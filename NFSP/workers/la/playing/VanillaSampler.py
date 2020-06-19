import numpy as np

from NFSP.workers.la.ParallelEnvs import ParallelEnvs
from NFSP.workers.la.playing.SamplingAlgo import SamplingAlgo, SeatActorBase
from PokerRL.rl import rl_util


class VanillaSampler(SamplingAlgo):

    def __init__(self, t_prof, env_bldr, br_buf2, avg_buf2, br_learner2, avg_learner2):
        super().__init__(t_prof=t_prof, env_bldr=env_bldr, br_buf2=br_buf2, avg_buf2=avg_buf2,
                         avg_learner2=avg_learner2, br_learner2=br_learner2, n_envs_br=t_prof.n_steps_per_iter_per_la,
                         n_envs_avg=t_prof.n_steps_per_iter_per_la)
        self._parallel_env = ParallelEnvs(t_prof=t_prof, env_bldr=env_bldr, n_envs=t_prof.n_steps_per_iter_per_la)

        self._last_step_wrappers = self._parallel_env.reset()

        self._seat_actors = [
            _SeatActor(t_prof=t_prof, env_bldr=self._env_bldr, seat_id=p,
                       br_memory_savers=self._br_memory_savers[p], avg_buf_savers=self._avg_memory_savers[p],
                       br_learner=br_learner2[p], avg_learner=avg_learner2[p], sampler=self)
            for p in range(self._env_bldr.N_SEATS)
        ]

        for p in range(env_bldr.N_SEATS):
            self._seat_actors[p].init(sws=[sw for plyr_sws in self._last_step_wrappers for sw in plyr_sws], nfsp_iter=0)

    def play(self, nfsp_iter):
        assert self._t_prof.n_steps_per_iter_per_la % self._parallel_env.n_envs == 0
        for n in range(self._t_prof.n_steps_per_iter_per_la // self._parallel_env.n_envs):

            # merge player's lists
            sws = [sw for plyr_sws in self._last_step_wrappers for sw in plyr_sws]

            # Both players must see all envs here
            for s in self._seat_actors:
                s.update_if_terminal(step_wrappers=sws, nfsp_iter=nfsp_iter)

            # Let players act on the envs
            for s in self._seat_actors:
                s.act(self._last_step_wrappers[s.seat_id])

            # Step envs
            self._last_step_wrappers = self._parallel_env.step(step_wraps=sws)


class _SeatActor:

    def __init__(self, seat_id, t_prof, env_bldr, br_memory_savers, avg_buf_savers, br_learner, avg_learner, sampler):
        self.seat_id = seat_id

        self._t_prof = t_prof
        self._env_bldr = env_bldr

        self.br_learner = br_learner
        self.avg_learner = avg_learner

        self.sampler = sampler

        self._avg_buf_savers = avg_buf_savers
        self._br_memory_savers = br_memory_savers

        self._current_policy_tags = None

    def init(self, sws, nfsp_iter):
        self._current_policy_tags = np.empty(shape=self._t_prof.n_steps_per_iter_per_la, dtype=np.int32)

        for sw in sws:
            self._current_policy_tags[sw.env_idx] = SeatActorBase.pick_training_policy(br_prob=self.sampler.antic)
            self._br_memory_savers[sw.env_idx].reset(range_idx=sw.range_idxs[self.seat_id])
            self._avg_buf_savers[sw.env_idx].reset(range_idx=sw.range_idxs[self.seat_id],
                                                   sample_weight=nfsp_iter if self._t_prof.linear else 1
                                                   )

    def act(self, step_wrappers):
        # """""""""""""""""""""
        # Act
        # """""""""""""""""""""
        SeatActorBase.act_mixed(step_wrappers=step_wrappers, owner=self.seat_id, br_learner=self.br_learner,
                                avg_learner=self.avg_learner, current_policy_tags=self._current_policy_tags,
                                random_prob=self.br_learner.eps)

        # """""""""""""""""""""
        # Add to memories
        # """""""""""""""""""""
        for sw in step_wrappers:
            e_i = sw.env_idx
            if (self._current_policy_tags[e_i] == SeatActorBase.BR) and (
                    self._t_prof.add_random_actions_to_buffer or (not sw.action_was_random)):
                self._avg_buf_savers[e_i].add_step(pub_obs=sw.obs,
                                                   a=sw.action,
                                                   legal_actions_mask=rl_util.get_legal_action_mask_np(
                                                       n_actions=self._env_bldr.N_ACTIONS,
                                                       legal_actions_list=sw.legal_actions_list)
                                                   )
            self._br_memory_savers[e_i].add_experience(obs_t_before_acted=sw.obs,
                                                       a_selected_t=sw.action,
                                                       legal_actions_list_t=sw.legal_actions_list)

    def update_if_terminal(self, step_wrappers, nfsp_iter):
        for sw in step_wrappers:
            if sw.TERMINAL:
                self._br_memory_savers[sw.env_idx].add_to_buf(
                    reward_p=sw.term_rew_all[self.seat_id],
                    terminal_obs=sw.term_obs,
                )
                self._br_memory_savers[sw.env_idx].reset(range_idx=sw.range_idxs[self.seat_id])
                self._avg_buf_savers[sw.env_idx].reset(range_idx=sw.range_idxs[self.seat_id],
                                                       sample_weight=nfsp_iter if self._t_prof.linear else 1
                                                       )

                self._current_policy_tags[sw.env_idx] = SeatActorBase.pick_training_policy(br_prob=self.sampler.antic)
