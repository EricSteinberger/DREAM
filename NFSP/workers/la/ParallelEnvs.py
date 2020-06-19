import copy

import numpy as np


class ParallelEnvs:

    def __init__(self, t_prof, env_bldr, n_envs):
        self._t_prof = t_prof
        self._env_bldr = env_bldr
        self.n_envs = n_envs
        self._env_wrappers = [
            self._env_bldr.get_new_wrapper(is_evaluating=False)
            for _ in range(n_envs)
        ]

    def reset(self):
        step_wraps = [
            []
            for _ in range(self._t_prof.n_seats)
        ]
        for e_idx, ew in enumerate(self._env_wrappers):
            obs, rew_all, done, info = ew.reset()
            step_wraps[ew.env.current_player.seat_id].append(
                StepWrapper(
                    obs=obs,
                    range_idxs=[ew.env.get_range_idx(p_id=p) for p in range(self._t_prof.n_seats)],
                    legal_actions_list=ew.env.get_legal_actions(),
                    env_idx=e_idx,
                )
            )

        return step_wraps

    def step(self, step_wraps):
        new_step_wraps = [[] for _ in range(self._t_prof.n_seats)]

        actions = np.empty(self.n_envs, np.int32)
        for sw in step_wraps:
            actions[sw.env_idx] = sw.action

        for e_idx, ew in enumerate(self._env_wrappers):
            obs, rew_all, done, info = ew.step(actions[e_idx].item())

            if done:
                obs_n, rew_all_n, done_n, info_n = ew.reset()
                new_step_wraps[ew.env.current_player.seat_id].append(
                    TerminalStepWrapper(
                        term_rew_all=copy.deepcopy(rew_all),
                        term_obs=obs,

                        obs=obs_n,
                        range_idxs=[ew.env.get_range_idx(p_id=p) for p in range(self._t_prof.n_seats)],
                        legal_actions_list=ew.env.get_legal_actions(),
                        env_idx=e_idx,
                    )
                )

            else:
                new_step_wraps[ew.env.current_player.seat_id].append(
                    StepWrapper(
                        obs=obs,
                        range_idxs=[ew.env.get_range_idx(p_id=p) for p in range(self._t_prof.n_seats)],
                        legal_actions_list=ew.env.get_legal_actions(),
                        env_idx=e_idx,
                    )
                )
        return new_step_wraps

    def state_dict(self):
        return {
            "states": [ew.state_dict() for ew in self._env_wrappers]
        }

    def load_state_dict(self, state):
        for e_idx, ew in enumerate(self._env_wrappers):
            ew.load_state_dict(state["states"][e_idx])


class StepWrapper:
    TERMINAL = False

    def __init__(self,
                 env_idx,
                 obs, range_idxs, legal_actions_list):
        self.obs = obs
        self.range_idxs = range_idxs
        self.legal_actions_list = legal_actions_list

        self.env_idx = env_idx

        self.action = None
        self.action_was_random = None


class TerminalStepWrapper:
    TERMINAL = True

    def __init__(self,
                 term_rew_all, term_obs,
                 env_idx,
                 obs, range_idxs, legal_actions_list,
                 ):
        self.term_rew_all = term_rew_all
        self.term_obs = term_obs

        self.env_idx = env_idx

        self.obs = obs
        self.range_idxs = range_idxs
        self.legal_actions_list = legal_actions_list

        self.action = None
        self.action_was_random = None
