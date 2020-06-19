# Copyright (c) Eric Steinberger 2020

import numpy as np
import torch

from DREAM_and_DeepCFR.workers.la.sampling_algorithms._SamplerBase import SamplerBase as _SamplerBase
from PokerRL.rl import rl_util


class MultiOutcomeSampler(_SamplerBase):
    """
    How to get to next state:
        -   Each time ""traverser"" acts, a number of sub-trees are followed. For each sample, the remaining deck is
            reshuffled to ensure a random future.

        -   When any other player acts, 1 action is chosen w.r.t. their strategy.

        -   When the environment acts, 1 action is chosen according to its natural dynamics. Note that the PokerRL
            environment does this inherently, which is why there is no code for that in this class.


    When what is stored to where:
        -   At every time a player other than ""traverser"" acts, we store their action probability vector to their
            reservoir buffer.

        -   Approximate immediate regrets are stored to ""traverser""'s advantage buffer at every node at which they
            act.
    """

    def __init__(self,
                 env_bldr,
                 adv_buffers,
                 avrg_buffers=None,
                 n_actions_traverser_samples=3,
                 after_x_only_one=None,
                 eps=0.5,
                 ):
        """
        Args:
            env_bldr:
            adv_buffers:
            avrg_buffers:
            n_actions_traverser_samples (int):  None:     Equivalent to External Sampling (ES)
                                                1:        Equivalent to Outcome Sampling (OS) with uniform policy
                                                between:  A blend between ES and OS

            after_x_only_one (int):            In long games, MOS with >1 actions might be too expensive.
                                               To aid this, ""after_x_only_one"" allows you to specify after how many
                                               branchings only one action is sampled.

        """
        if eps > 0:
            assert avrg_buffers is None
        super().__init__(env_bldr=env_bldr, adv_buffers=adv_buffers, avrg_buffers=avrg_buffers)
        self._eps = eps
        self._n_actions_traverser_samples = n_actions_traverser_samples
        self._depth_after_which_one = after_x_only_one

        self._actions_arranged = np.arange(self._env_bldr.N_ACTIONS)
        self.total_node_count_traversed = 0

    def _get_n_a_to_sample(self, trav_depth, n_legal_actions):
        if (self._depth_after_which_one is not None) and (trav_depth > self._depth_after_which_one):
            return 1
        if self._n_actions_traverser_samples is None:
            return n_legal_actions
        return min(self._n_actions_traverser_samples, n_legal_actions)

    def _traverser_act(self, start_state_dict, traverser, trav_depth, sample_reach, plyrs_range_idxs, iteration_strats,
                       cfr_iter):
        self.total_node_count_traversed += 1

        """
        Last state values are the average, not the sum of all samples of that state since we add
        v~(I) = * p(a) * |A(I)|. Since we sample multiple actions on each traverser node, we have to average over
        their returns like: v~(I) * Sum_a=0_N (v~(I|a) * p(a) * ||A(I)|| / N).
        """
        self._env_wrapper.load_state_dict(start_state_dict)
        legal_actions_list = self._env_wrapper.env.get_legal_actions()
        legal_action_mask = rl_util.get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                                legal_actions_list=legal_actions_list,
                                                                device=self._adv_buffers[traverser].device,
                                                                dtype=torch.float32)
        current_pub_obs = self._env_wrapper.get_current_obs()

        traverser_range_idx = plyrs_range_idxs[traverser]

        strat_i = iteration_strats[traverser].get_a_probs(
            pub_obses=[current_pub_obs],
            range_idxs=[traverser_range_idx],
            legal_actions_lists=[legal_actions_list],
            to_np=True
        )[0]

        n_legal_actions = len(legal_actions_list)
        n_actions_to_smpl = self._get_n_a_to_sample(trav_depth=trav_depth, n_legal_actions=n_legal_actions)
        sample_strat = (1 - self._eps) * strat_i + self._eps * (legal_action_mask.cpu().numpy() / n_legal_actions)

        # """""""""""""""""""""""""
        # Create next states
        # """""""""""""""""""""""""
        cumm_rew = 0.0
        aprx_imm_reg = torch.zeros(size=(self._env_bldr.N_ACTIONS,),
                                   dtype=torch.float32,
                                   device=self._adv_buffers[traverser].device)
        _1_over_sample_reach_sum = 0
        _sample_reach_sum = 0
        for _c, a in enumerate(np.random.choice(self._actions_arranged, p=sample_strat, size=n_actions_to_smpl)):
            # Re-initialize environment after one action-branch loop finished with current state and random future
            if _c > 0:
                self._env_wrapper.load_state_dict(start_state_dict)
                self._env_wrapper.env.reshuffle_remaining_deck()

            _obs, _rew_for_all, _done, _info = self._env_wrapper.step(a)
            _u_a = _rew_for_all[traverser]

            _sample_reach = sample_reach * sample_strat[a] * n_legal_actions
            _sample_reach_sum += _sample_reach
            _1_over_sample_reach_sum += 1 / _sample_reach
            # Recursion over sub-trees
            if not _done:
                _u_a += self._recursive_traversal(start_state_dict=self._env_wrapper.state_dict(),
                                                  traverser=traverser,
                                                  trav_depth=trav_depth + 1,
                                                  plyrs_range_idxs=plyrs_range_idxs,
                                                  iteration_strats=iteration_strats,
                                                  sample_reach=_sample_reach,
                                                  cfr_iter=cfr_iter)

            # accumulate reward for backward-pass on tree
            cumm_rew += strat_i[a] / sample_strat[a] * _u_a

            # """"""""""""""""""""""""
            # Compute the approximate
            # immediate regret
            # """"""""""""""""""""""""

            _aprx_imm_reg = torch.full_like(aprx_imm_reg, fill_value=-strat_i[a] * _u_a)  # This is for all actions != a
            _aprx_imm_reg[a] += _u_a  # add regret for a and undo the change made to a's regret in the line above.
            aprx_imm_reg += _aprx_imm_reg / _sample_reach

        aprx_imm_reg *= legal_action_mask / _1_over_sample_reach_sum  # mean over all legal actions sampled

        # add current datapoint to ADVBuf
        self._adv_buffers[traverser].add(pub_obs=current_pub_obs,
                                         range_idx=traverser_range_idx,
                                         legal_action_mask=legal_action_mask,
                                         adv=aprx_imm_reg,
                                         iteration=(cfr_iter + 1) / _sample_reach_sum,
                                         )

        # /n_actions_to_smpl  because we summed that many returns and want their mean
        return cumm_rew / n_actions_to_smpl

    def _any_non_traverser_act(self, start_state_dict, traverser, plyrs_range_idxs, trav_depth, iteration_strats,
                               sample_reach, cfr_iter):
        self.total_node_count_traversed += 1
        return super()._any_non_traverser_act(start_state_dict, traverser, plyrs_range_idxs, trav_depth,
                                              iteration_strats,
                                              sample_reach, cfr_iter)
