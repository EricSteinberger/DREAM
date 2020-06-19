# Copyright (c) Eric Steinberger 2020

import numpy as np
import torch

from DREAM_and_DeepCFR.workers.la.sampling_algorithms._SamplerBase import SamplerBase as _SamplerBase
from PokerRL.rl import rl_util


class ExternalSampler(_SamplerBase):
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
        # self._reg_buf = None
        super().__init__(env_bldr=env_bldr, adv_buffers=adv_buffers, avrg_buffers=avrg_buffers)

        self._actions_arranged = np.arange(self._env_bldr.N_ACTIONS)

    def generate(self, n_traversals, traverser, iteration_strats, cfr_iter, ):
        # self._reg_buf = [[] for _ in range(self._env_bldr.rules.N_CARDS_IN_DECK)]

        super().generate(n_traversals, traverser, iteration_strats, cfr_iter)
        # if traverser == 0:
        #     print("STD:  ", np.sum(np.array([np.array(x).std(axis=0) for x in self._reg_buf]), axis=0))
        #     print("Mean: ", np.sum(np.array([np.array(x).mean(axis=0) for x in self._reg_buf]), axis=0))

    def _traverser_act(self, start_state_dict, traverser, trav_depth, sample_reach, plyrs_range_idxs, iteration_strats,
                       cfr_iter):
        """
        Last state values are the average, not the sum of all samples of that state since we add
        v~(I) = * p(a) * |A(I)|. Since we sample multiple actions on each traverser node, we have to average over
        their returns like: v~(I) * Sum_a=0_N (v~(I|a) * p(a) * ||A(I)|| / N).
        """
        self.total_node_count_traversed += 1
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
            to_np=False
        )[0]

        u = torch.zeros(size=(self._env_bldr.N_ACTIONS,),
                        dtype=torch.float32,
                        device=self._adv_buffers[traverser].device)
        for _c, a in enumerate(legal_actions_list):
            # Re-initialize environment after one action-branch loop finished with current state and random future
            if _c > 0:
                self._env_wrapper.load_state_dict(start_state_dict)
                self._env_wrapper.env.reshuffle_remaining_deck()

            _obs, _rew_for_all, _done, _info = self._env_wrapper.step(a)
            if _done:
                self.total_node_count_traversed += 1
                u[a] = _rew_for_all[traverser]
            else:
                # Recursion over sub-trees
                u[a] += self._recursive_traversal(start_state_dict=self._env_wrapper.state_dict(),
                                                  traverser=traverser,
                                                  trav_depth=trav_depth + 1,
                                                  plyrs_range_idxs=plyrs_range_idxs,
                                                  iteration_strats=iteration_strats,
                                                  sample_reach=None,
                                                  cfr_iter=cfr_iter)

        # """"""""""""""""""""""""
        # Compute the approximate
        # immediate regret
        # """"""""""""""""""""""""
        v = (strat_i * u).sum().item()
        aprx_imm_reg = torch.full_like(u, fill_value=-v)
        aprx_imm_reg += u
        aprx_imm_reg *= legal_action_mask

        # add current datapoint to ADVBuf
        self._adv_buffers[traverser].add(pub_obs=current_pub_obs,
                                         range_idx=traverser_range_idx,
                                         legal_action_mask=legal_action_mask,
                                         adv=aprx_imm_reg,
                                         iteration=cfr_iter + 1,
                                         )

        # if trav_depth == 0 and traverser == 0:
        #     self._reg_buf[traverser_range_idx].append(aprx_imm_reg.clone().cpu().numpy())
        return v

    def _any_non_traverser_act(self, start_state_dict, traverser, plyrs_range_idxs, trav_depth, iteration_strats,
                               sample_reach, cfr_iter):
        self.total_node_count_traversed += 1
        return super()._any_non_traverser_act(start_state_dict, traverser, plyrs_range_idxs, trav_depth,
                                              iteration_strats,
                                              sample_reach, cfr_iter)
