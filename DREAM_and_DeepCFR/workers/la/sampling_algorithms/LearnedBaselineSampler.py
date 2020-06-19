# Copyright (c) Eric Steinberger 2020

import numpy as np
import torch

from DREAM_and_DeepCFR.workers.la.sampling_algorithms._SamplerBase import SamplerBase
from PokerRL.rl import rl_util


class LearnedBaselineSampler(SamplerBase):
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
                 baseline_net,
                 baseline_buf,
                 eps=0.5,
                 avrg_buffers=None,
                 ):
        super().__init__(env_bldr=env_bldr, adv_buffers=adv_buffers, avrg_buffers=avrg_buffers)
        self._baseline_net = baseline_net
        self._baseline_buf = baseline_buf

        # self._reg_buf = None

        self._eps = eps
        self._actions_arranged = np.arange(self._env_bldr.N_ACTIONS)

        self.total_node_count_traversed = 0

    def generate(self, n_traversals, traverser, iteration_strats, cfr_iter, ):
        # self._reg_buf = [[] for _ in range(self._env_bldr.rules.N_CARDS_IN_DECK)]

        super().generate(n_traversals, traverser, iteration_strats, cfr_iter)
        # if traverser == 0:
        #     print("STD:  ", np.sum(np.array([np.array(x).std(axis=0) for x in self._reg_buf]), axis=0))
        #     print("Mean: ", np.sum(np.array([np.array(x).mean(axis=0) for x in self._reg_buf]), axis=0))

    def _traverser_act(self, start_state_dict, traverser, trav_depth, plyrs_range_idxs, iteration_strats, sample_reach,
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
        pub_obs_t = self._env_wrapper.get_current_obs()
        traverser_range_idx = plyrs_range_idxs[traverser]

        # """""""""""""""""""""""""
        # Strategy
        # """""""""""""""""""""""""
        strat_i = iteration_strats[traverser].get_a_probs(
            pub_obses=[pub_obs_t],
            range_idxs=[traverser_range_idx],
            legal_actions_lists=[legal_actions_list],
            to_np=False,
        )[0]

        # """""""""""""""""""""""""
        # Sample action
        # """""""""""""""""""""""""
        n_legal_actions = len(legal_actions_list)
        sample_strat = (1 - self._eps) * strat_i + self._eps * (legal_action_mask.cpu() / n_legal_actions)
        a = torch.multinomial(sample_strat.cpu(), num_samples=1).item()

        # Step
        pub_obs_tp1, rew_for_all, done, _info = self._env_wrapper.step(a)
        legal_action_mask_tp1 = rl_util.get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                                    legal_actions_list=self._env_wrapper.env.get_legal_actions(),
                                                                    device=self._adv_buffers[traverser].device,
                                                                    dtype=torch.float32)

        # """""""""""""""""""""""""
        # Recursion
        # """""""""""""""""""""""""
        if done:
            strat_tp1 = torch.zeros_like(strat_i)
        else:
            u_bootstrap, strat_tp1 = self._recursive_traversal(
                start_state_dict=self._env_wrapper.state_dict(),
                traverser=traverser,
                trav_depth=trav_depth + 1,
                plyrs_range_idxs=plyrs_range_idxs,
                iteration_strats=iteration_strats,
                cfr_iter=cfr_iter,
                sample_reach=sample_reach * sample_strat[a] * n_legal_actions
            )

        # """""""""""""""""""""""""
        # Utility
        # """""""""""""""""""""""""
        utility = self._get_utility(
            traverser=traverser,
            u_bootstrap=rew_for_all[traverser] if done else u_bootstrap,
            range_idx_crazy_embedded=_crazy_embed(plyrs_range_idxs=plyrs_range_idxs),
            pub_obs=pub_obs_t,
            legal_actions_list=legal_actions_list,
            legal_action_mask=legal_action_mask,
            a=a,
            sample_strat=sample_strat,
        )

        # Regret
        aprx_imm_reg = torch.full(size=(self._env_bldr.N_ACTIONS,),
                                  fill_value=-(utility * strat_i).sum(),
                                  dtype=torch.float32,
                                  device=self._adv_buffers[traverser].device)
        aprx_imm_reg += utility
        aprx_imm_reg *= legal_action_mask

        # add current datapoint to ADVBuf
        self._adv_buffers[traverser].add(pub_obs=pub_obs_t,
                                         range_idx=traverser_range_idx,
                                         legal_action_mask=legal_action_mask,
                                         adv=aprx_imm_reg,
                                         iteration=(cfr_iter + 1) / sample_reach,
                                         )

        # add datapoint to baseline net
        self._baseline_buf.add(
            pub_obs=pub_obs_t,
            range_idx_crazy_embedded=_crazy_embed(plyrs_range_idxs=plyrs_range_idxs),
            legal_action_mask=legal_action_mask,
            r=rew_for_all[0],
            a=a,
            done=done,
            pub_obs_tp1=pub_obs_tp1,
            strat_tp1=strat_tp1,
            legal_action_mask_tp1=legal_action_mask_tp1,
        )

        # if trav_depth == 0 and traverser == 0:
        #     self._reg_buf[traverser_range_idx].append(aprx_imm_reg.clone().cpu().numpy())

        return (utility * strat_i).sum(), strat_i

    def _any_non_traverser_act(self, start_state_dict, traverser, plyrs_range_idxs, trav_depth, iteration_strats,
                               sample_reach, cfr_iter):
        self.total_node_count_traversed += 1
        self._env_wrapper.load_state_dict(start_state_dict)
        p_id_acting = self._env_wrapper.env.current_player.seat_id

        current_pub_obs = self._env_wrapper.get_current_obs()
        range_idx = plyrs_range_idxs[p_id_acting]
        legal_actions_list = self._env_wrapper.env.get_legal_actions()
        legal_action_mask = rl_util.get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                                legal_actions_list=legal_actions_list,
                                                                device=self._adv_buffers[traverser].device,
                                                                dtype=torch.float32)
        # """""""""""""""""""""""""
        # The players strategy
        # """""""""""""""""""""""""
        strat_opp = iteration_strats[p_id_acting].get_a_probs(
            pub_obses=[current_pub_obs],
            range_idxs=[range_idx],
            legal_actions_lists=[legal_actions_list],
            to_np=False
        )[0]

        # """""""""""""""""""""""""
        # Execute action from strat
        # """""""""""""""""""""""""
        a = torch.multinomial(strat_opp.cpu(), num_samples=1).item()
        pub_obs_tp1, rew_for_all, done, _info = self._env_wrapper.step(a)
        legal_action_mask_tp1 = rl_util.get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                                    legal_actions_list=self._env_wrapper.env.get_legal_actions(),
                                                                    device=self._adv_buffers[traverser].device,
                                                                    dtype=torch.float32)

        # """""""""""""""""""""""""
        # Adds to opponent's
        # average buffer if
        # applicable
        # """""""""""""""""""""""""
        if self._avrg_buffers is not None:
            self._avrg_buffers[p_id_acting].add(
                pub_obs=current_pub_obs,
                range_idx=range_idx,
                legal_actions_list=legal_actions_list,
                a_probs=strat_opp.to(self._avrg_buffers[p_id_acting].device).squeeze(),
                iteration=(cfr_iter + 1) / sample_reach
            )

        # """""""""""""""""""""""""
        # Recursion
        # """""""""""""""""""""""""
        if done:
            strat_tp1 = torch.zeros_like(strat_opp)
            self.total_node_count_traversed += 1
        else:
            u_bootstrap, strat_tp1 = self._recursive_traversal(
                start_state_dict=self._env_wrapper.state_dict(),
                traverser=traverser,
                trav_depth=trav_depth + 1,
                plyrs_range_idxs=plyrs_range_idxs,
                iteration_strats=iteration_strats,
                cfr_iter=cfr_iter,
                sample_reach=sample_reach
            )

        # """""""""""""""""""""""""
        # Utility
        # """""""""""""""""""""""""
        utility = self._get_utility(
            traverser=traverser,
            u_bootstrap=rew_for_all[traverser] if done else u_bootstrap,
            pub_obs=current_pub_obs,
            range_idx_crazy_embedded=_crazy_embed(plyrs_range_idxs=plyrs_range_idxs),
            legal_actions_list=legal_actions_list,
            legal_action_mask=legal_action_mask,
            a=a,
            sample_strat=strat_opp,
        )

        # add datapoint to baseline net
        self._baseline_buf.add(
            pub_obs=current_pub_obs,
            range_idx_crazy_embedded=_crazy_embed(plyrs_range_idxs=plyrs_range_idxs),
            legal_action_mask=legal_action_mask,
            r=rew_for_all[0],  # 0 bc we mirror for 1... zero-sum
            a=a,
            done=done,

            pub_obs_tp1=pub_obs_tp1,
            strat_tp1=strat_tp1,
            legal_action_mask_tp1=legal_action_mask_tp1,
        )

        return (utility * strat_opp).sum(), strat_opp

    def _get_utility(self, traverser, pub_obs, range_idx_crazy_embedded,
                     legal_actions_list, legal_action_mask, u_bootstrap, a, sample_strat):

        ######################
        # Remove variance from
        # action
        ######################
        baselines = self._baseline_net.get_b(
            pub_obses=[pub_obs],
            range_idxs=[range_idx_crazy_embedded],
            legal_actions_lists=[legal_actions_list],
            to_np=False,
        )[0] * (1 if traverser == 0 else -1)

        # print(baselines[a], u_bootstrap, a)
        utility = baselines * legal_action_mask
        utility[a] += (u_bootstrap - utility[a]) / sample_strat[a]

        return utility


# See MPM_Baseline
def _crazy_embed(plyrs_range_idxs):
    return plyrs_range_idxs[0] * 10000 + plyrs_range_idxs[1]
