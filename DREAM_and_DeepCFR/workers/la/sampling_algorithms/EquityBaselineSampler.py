import numpy as np
import torch

from DREAM_and_DeepCFR.workers.la.sampling_algorithms._SamplerBase import SamplerBase
from PokerRL import Poker, StandardLeduc
from PokerRL.rl import rl_util

CANCEL_BOARD = False
CALL_AND_RAISE_ZERO = False


class VR_OS_Sampler(SamplerBase):
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
                 eps=0.5,
                 ):
        super().__init__(env_bldr=env_bldr, adv_buffers=adv_buffers, avrg_buffers=avrg_buffers)

        # self._reg_buf = None
        self._eps = eps
        self._turn_off_baseline = turn_off_baseline
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
        current_pub_obs = self._env_wrapper.get_current_obs()
        round_t = self._env_wrapper.env.current_round
        traverser_range_idx = plyrs_range_idxs[traverser]

        # """""""""""""""""""""""""
        # Strategy
        # """""""""""""""""""""""""
        strat_i = iteration_strats[traverser].get_a_probs(
            pub_obses=[current_pub_obs],
            range_idxs=[traverser_range_idx],
            legal_actions_lists=[legal_actions_list],
            to_np=False,
        )[0]

        # """""""""""""""""""""""""
        # Sample action
        # """""""""""""""""""""""""
        n_legal_actions = len(legal_actions_list)
        sample_strat = (1 - self._eps) * strat_i + self._eps * (legal_action_mask.cpu() / n_legal_actions)
        a = np.random.choice(self._actions_arranged, p=sample_strat.numpy())

        # Step
        pub_obs_tp1, rew_for_all, done, _info = self._env_wrapper.step(a)
        round_tp1 = self._env_wrapper.env.current_round

        # """""""""""""""""""""""""
        # Utility
        # """""""""""""""""""""""""
        utility = self._get_utility(
            traverser=traverser,
            acting_player=traverser,
            u_bootstrap=rew_for_all[traverser] if done else self._recursive_traversal(
                start_state_dict=self._env_wrapper.state_dict(),
                traverser=traverser,
                trav_depth=trav_depth + 1,
                plyrs_range_idxs=plyrs_range_idxs,
                iteration_strats=iteration_strats,
                cfr_iter=cfr_iter,
                sample_reach=sample_reach * sample_strat[a] * n_legal_actions
            ),  # recursion
            pub_obs=current_pub_obs,
            range_idx_trav=plyrs_range_idxs[traverser],
            range_idx_opp=plyrs_range_idxs[1 - traverser],
            legal_actions_list=legal_actions_list,
            legal_action_mask=legal_action_mask,
            a=a,
            round_t=round_t,
            round_tp1=round_tp1,
            sample_strat=sample_strat,
            pub_obs_tp1=pub_obs_tp1,
        )

        # Regret
        aprx_imm_reg = torch.full(size=(self._env_bldr.N_ACTIONS,),
                                  fill_value=-(utility * strat_i).sum(),
                                  dtype=torch.float32,
                                  device=self._adv_buffers[traverser].device)
        aprx_imm_reg += utility
        aprx_imm_reg *= legal_action_mask

        # add current datapoint to ADVBuf
        self._adv_buffers[traverser].add(pub_obs=current_pub_obs,
                                         range_idx=traverser_range_idx,
                                         legal_action_mask=legal_action_mask,
                                         adv=aprx_imm_reg,
                                         iteration=(cfr_iter + 1) / sample_reach,
                                         )

        # if trav_depth == 0 and traverser == 0:
        #     self._reg_buf[traverser_range_idx].append(aprx_imm_reg.clone().cpu().numpy())
        return (utility * strat_i).sum()

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
        round_t = self._env_wrapper.env.current_round

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
                iteration=cfr_iter + 1)

        # """""""""""""""""""""""""
        # Execute action from strat
        # """""""""""""""""""""""""
        a = torch.multinomial(strat_opp.cpu(), num_samples=1).item()
        pub_obs_tp1, rew_for_all, done, _info = self._env_wrapper.step(a)
        round_tp1 = self._env_wrapper.env.current_round

        # """""""""""""""""""""""""
        # Utility
        # """""""""""""""""""""""""
        utility = self._get_utility(
            traverser=traverser,
            acting_player=1 - traverser,
            u_bootstrap=rew_for_all[traverser] if done else self._recursive_traversal(
                start_state_dict=self._env_wrapper.state_dict(),
                traverser=traverser,
                trav_depth=trav_depth,
                plyrs_range_idxs=plyrs_range_idxs,
                iteration_strats=iteration_strats,
                cfr_iter=cfr_iter,
                sample_reach=sample_reach,
            ),
            pub_obs=current_pub_obs,
            range_idx_trav=plyrs_range_idxs[traverser],
            range_idx_opp=plyrs_range_idxs[1 - traverser],
            legal_actions_list=legal_actions_list,
            legal_action_mask=legal_action_mask,
            a=a,
            round_t=round_t,
            round_tp1=round_tp1,
            sample_strat=strat_opp,
            pub_obs_tp1=pub_obs_tp1,
        )

        return (utility * strat_opp).sum()

    # This again exploits leduc A LOT.
    def _get_utility(self, traverser, acting_player, pub_obs, pub_obs_tp1, range_idx_trav, range_idx_opp,
                     legal_actions_list, legal_action_mask,
                     u_bootstrap, a, sample_strat, round_t, round_tp1):
        if self._turn_off_baseline:
            u = torch.zeros((self._env_bldr.N_ACTIONS,), dtype=torch.float32)
            u[a] = u_bootstrap / sample_strat[a]
            return u

        ######################
        # Remove variance from
        # flop.
        # Exploits that only
        # one board card was
        # dealt!
        ######################
        if round_tp1 > round_t and CANCEL_BOARD:
            flop_probs = torch.tensor([2, 2, 2], dtype=torch.float32)
            flop_probs[range_idx_trav // self._env_bldr.rules.N_SUITS] -= 1
            flop_probs[range_idx_opp // self._env_bldr.rules.N_SUITS] -= 1
            flop_probs /= flop_probs.sum()
            card_rank_actually_dealt = np.argmax(pub_obs_tp1[22:25]).item()

            half_the_pot = self._get_pot_after_action(pub_obs=pub_obs, a=a, round_t=round_t)

            flop_utility = torch.tensor([
                half_the_pot * self._get_equity(board_card_rank=rank,
                                                rank_trav=range_idx_trav // self._env_bldr.rules.N_SUITS,
                                                rank_opp=range_idx_opp // self._env_bldr.rules.N_SUITS)
                for rank, p in enumerate(flop_probs)
            ], dtype=torch.float32)
            flop_utility[card_rank_actually_dealt] += (u_bootstrap - flop_utility[card_rank_actually_dealt]) \
                                                      / flop_probs[card_rank_actually_dealt]
            u_bootstrap = (flop_utility * flop_probs).sum()

        ######################
        # Remove variance from
        # action
        ######################
        baselines = self._get_rollout_baselines(acting_player=acting_player, traverser=traverser, pub_obs=pub_obs,
                                                rank_trav=range_idx_trav // self._env_bldr.rules.N_SUITS,
                                                rank_opp=range_idx_opp // self._env_bldr.rules.N_SUITS,
                                                round_t=round_t,
                                                legal_actions_list=legal_actions_list)

        if CALL_AND_RAISE_ZERO:
            baselines[Poker.BET_RAISE] = 0
            baselines[Poker.CHECK_CALL] = 0
            # baselines[Poker.FOLD] = 0

        utility = baselines * legal_action_mask
        utility[a] += (u_bootstrap - utility[a]) / sample_strat[a]

        return utility

    def _get_rollout_baselines(self, acting_player, traverser, round_t, pub_obs, rank_trav, rank_opp,
                               legal_actions_list):
        # self._env_wrapper.print_obs(pub_obs)
        board_card_rank = Poker.CARD_NOT_DEALT_TOKEN_1D if round_t == 0 else np.argmax(pub_obs[22:25]).item()
        u = torch.zeros((self._env_bldr.N_ACTIONS,), dtype=torch.float32)
        for a in legal_actions_list:
            P = self._get_pot_after_action(pub_obs=pub_obs, a=a, round_t=round_t)
            if a == Poker.FOLD:
                u[a] = P * (-1 if traverser == acting_player else 1)
            else:
                u[a] = P * self._get_equity(board_card_rank=board_card_rank, rank_trav=rank_trav, rank_opp=rank_opp)
        return u

    def _get_pot_after_action(self, pub_obs, a, round_t):
        pot = pub_obs[4]
        bets_out = [pub_obs[17], pub_obs[20]]
        next_bet_size = (
                            StandardLeduc.BIG_BET
                            if round_t >= StandardLeduc.ROUND_WHERE_BIG_BET_STARTS
                            else StandardLeduc.SMALL_BET) \
                        / self._env_wrapper.env.REWARD_SCALAR
        if a == Poker.FOLD:
            return pot / 2 + min(bets_out)
        elif a == Poker.CHECK_CALL:
            return pot / 2 + max(bets_out)
        elif a == Poker.BET_RAISE:
            return pot / 2 + max(bets_out) + next_bet_size

    def _get_equity(self, board_card_rank, rank_trav, rank_opp):
        """
        between -1 and 1. equity from perspective of traverser
        """
        if board_card_rank == Poker.CARD_NOT_DEALT_TOKEN_1D:
            board_probs = torch.tensor([2, 2, 2], dtype=torch.float32)
            board_probs[rank_trav] -= 1
            board_probs[rank_opp] -= 1
            board_probs /= board_probs.sum()
            return sum([
                p * self._get_equity(board_card_rank=rank, rank_trav=rank_trav, rank_opp=rank_opp)
                for rank, p in enumerate(board_probs.tolist())
            ])
        else:
            if rank_trav == board_card_rank:
                return 1
            elif rank_opp == board_card_rank:
                return -1
            elif rank_trav > rank_opp:
                return 1
            elif rank_opp > rank_trav:
                return -1
            elif rank_trav == rank_opp:
                return 0
            else:
                raise NotImplementedError  # should cover all cases above
