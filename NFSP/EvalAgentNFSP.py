# Copyright (c) 2019 Eric Steinberger


import numpy as np

from NFSP.AvgWrapper import AvgWrapper
from PokerRL.rl.base_cls.EvalAgentBase import EvalAgentBase as _EvalAgentBase
from PokerRL.rl.errors import UnknownModeError


class EvalAgentNFSP(_EvalAgentBase):
    EVAL_MODE_AVG = "NFSP_Avg"
    ALL_MODES = [EVAL_MODE_AVG]

    def __init__(self, t_prof, mode=None, device=None):
        super().__init__(t_prof=t_prof, mode=mode, device=device)
        self.avg_args = t_prof.module_args["avg"]

        self.policies = [
            AvgWrapper(owner=p, env_bldr=self.env_bldr, avg_training_args=self.avg_args)
            for p in range(t_prof.n_seats)
        ]
        for pol in self.policies:
            pol.eval()

    def can_compute_mode(self):
        return True

    def get_a_probs_for_each_hand(self):
        """ BEFORE CALLING, NOTIFY EVALAGENT OF THE PAST ACTIONS / ACTIONSEQUENCE!!!!! """
        p_id_acting = self._internal_env_wrapper.env.current_player.seat_id

        if self._mode == self.EVAL_MODE_AVG:
            return self.policies[p_id_acting].get_a_probs_for_each_hand(
                pub_obs=self._internal_env_wrapper.get_current_obs(),
                legal_actions_list=self._internal_env_wrapper.env.get_legal_actions())

        else:
            raise UnknownModeError(self._mode)

    def get_a_probs(self):
        p_id_acting = self._internal_env_wrapper.env.current_player.seat_id
        range_idx = self._internal_env_wrapper.env.get_range_idx(p_id=p_id_acting)
        return self.policies[p_id_acting].get_a_probs(
            pub_obses=[self._internal_env_wrapper.get_current_obs()],
            range_idxs=np.array([range_idx], dtype=np.int32),
            legal_actions_lists=[self._internal_env_wrapper.env.get_legal_actions()]
        )[0]

    def get_action(self, step_env=True, need_probs=False):
        """ !! BEFORE CALLING, NOTIFY EVALAGENT OF THE PAST ACTIONS / ACTIONSEQUENCE !! """

        p_id_acting = self._internal_env_wrapper.env.current_player.seat_id
        range_idx = self._internal_env_wrapper.env.get_range_idx(p_id=p_id_acting)

        if self._mode == self.EVAL_MODE_AVG:
            if need_probs:  # only do if rly necessary
                a_probs_all_hands = self.get_a_probs_for_each_hand()
                a_probs = a_probs_all_hands[range_idx]
            else:
                a_probs_all_hands = None  # not needed
                a_probs = self.policies[p_id_acting].get_a_probs(
                    pub_obses=[self._internal_env_wrapper.get_current_obs()],
                    range_idxs=np.array([range_idx], dtype=np.int32),
                    legal_actions_lists=[self._internal_env_wrapper.env.get_legal_actions()]
                )[0]

            action = np.random.choice(np.arange(self.env_bldr.N_ACTIONS), p=a_probs)

            if step_env:
                self._internal_env_wrapper.step(action=action)

            return action, a_probs_all_hands

        else:
            raise UnknownModeError(self._mode)

    def update_weights(self, weights_for_eval_agent):
        for i in range(self.t_prof.n_seats):
            self.policies[i].load_net_state_dict(self.ray.state_dict_to_torch(weights_for_eval_agent[i],
                                                                              device=self.device))
            self.policies[i].eval()

    def _state_dict(self):
        return {
            "net_state_dicts": [pol.net_state_dict() for pol in self.policies],
        }

    def _load_state_dict(self, state_dict):
        for i in range(self.t_prof.n_seats):
            self.policies[i].load_net_state_dict(state_dict["net_state_dicts"][i])
