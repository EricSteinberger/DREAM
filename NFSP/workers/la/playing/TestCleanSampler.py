import numpy as np
import torch

from NFSP.workers.la.ParallelEnvs import ParallelEnvs
from NFSP.workers.la.playing.SamplingAlgo import SamplingAlgo, SeatActorBase
from PokerRL.rl import rl_util


class TestCleanSampler(SamplingAlgo):

    def __init__(self, t_prof, env_bldr, br_buf2, avg_buf2, br_learner2, avg_learner2):
        super().__init__(t_prof=t_prof, env_bldr=env_bldr, br_buf2=br_buf2, avg_buf2=avg_buf2,
                         avg_learner2=avg_learner2, br_learner2=br_learner2,
                         n_envs_br=t_prof.n_steps_br_per_iter_per_la,
                         n_envs_avg=t_prof.n_steps_avg_per_iter_per_la)
        self._parallel_env_br2 = [
            ParallelEnvs(t_prof=t_prof, env_bldr=env_bldr, n_envs=t_prof.n_steps_br_per_iter_per_la)
            for p in range(self._env_bldr.N_SEATS)
        ]

        self._parallel_env_avg2 = [
            ParallelEnvs(t_prof=t_prof, env_bldr=env_bldr, n_envs=t_prof.n_steps_avg_per_iter_per_la)
            for p in range(self._env_bldr.N_SEATS)
        ]

        self._last_step_wrappers_br2 = [
            E.reset()
            for E in self._parallel_env_br2
        ]

        self._last_step_wrappers_avg2 = [
            E.reset()
            for E in self._parallel_env_avg2
        ]

        self._seat_actors = [
            _SeatActor(t_prof=t_prof, env_bldr=self._env_bldr, seat_id=p,
                       br_memory_savers=self._br_memory_savers[p], avg_buf_savers=self._avg_memory_savers[p],
                       br_learner=br_learner2[p], avg_learner=avg_learner2[p], sampler=self)
            for p in range(self._env_bldr.N_SEATS)
        ]

        for p_id in range(env_bldr.N_SEATS):
            self._seat_actors[p_id].init(
                sws_br=[sw for plyr_sws in self._last_step_wrappers_br2[p_id] for sw in plyr_sws],
                sws_avg=[sw for plyr_sws in self._last_step_wrappers_avg2[p_id] for sw in plyr_sws],
                nfsp_iter=0
            )

    def play(self, nfsp_iter):
        for p in range(self._t_prof.n_seats):
            self._play_for_br(trav=p, n_steps=self._t_prof.n_steps_br_per_iter_per_la)
            self._play_for_avg(trav=p, n_steps=self._t_prof.n_steps_avg_per_iter_per_la, nfsp_iter=nfsp_iter)

    def _play_for_br(self, trav, n_steps):
        p_envs = self._parallel_env_br2[trav]
        for n in range(n_steps // p_envs.n_envs):
            # merge player's lists
            all_sws = [sw for plyr_sws in self._last_step_wrappers_br2[trav] for sw in plyr_sws]

            # Both players must see all envs here
            for s in self._seat_actors:
                s.update_if_terminal_for_br(all_sws, is_traverser=trav == s.owner)

            #####################
            # Players act
            #####################
            for s in self._seat_actors:
                if s.owner == trav:
                    s.act_for_br_trav(self._last_step_wrappers_br2[trav][s.owner])
                else:
                    s.act_for_br_opp(self._last_step_wrappers_br2[trav][s.owner])

            # Step envs
            self._last_step_wrappers_br2[trav] = p_envs.step(step_wraps=all_sws)

    def _play_for_avg(self, trav, n_steps, nfsp_iter):
        p_envs = self._parallel_env_avg2[trav]
        for n in range(n_steps // p_envs.n_envs):
            # merge player's lists
            all_sws = [sw for plyr_sws in self._last_step_wrappers_avg2[trav] for sw in plyr_sws]

            # Both players must see all envs here
            for s in self._seat_actors:
                s.update_if_terminal_for_avg(step_wrappers=all_sws, is_traverser=trav == s.owner, nfsp_iter=nfsp_iter)

            # Let players act on the envs
            for s in self._seat_actors:
                if s.owner == trav:
                    s.act_for_avg_trav(step_wrappers=self._last_step_wrappers_avg2[trav][s.owner])
                else:
                    s.act_for_avg_opp(step_wrappers=self._last_step_wrappers_avg2[trav][s.owner])

            # Step envs
            self._last_step_wrappers_avg2[trav] = p_envs.step(step_wraps=all_sws)


class _SeatActor:

    def __init__(self, seat_id, t_prof, env_bldr, br_memory_savers, avg_buf_savers, br_learner, avg_learner, sampler):
        self.owner = seat_id

        self._t_prof = t_prof
        self._env_bldr = env_bldr

        self.br_learner = br_learner
        self.avg_learner = avg_learner

        self.sampler = sampler

        self._n_actions_arranged = np.arange(self._env_bldr.N_ACTIONS)

        # For T_AVG
        self._avg_memory_savers = avg_buf_savers

        # For T_BR
        self._br_memory_savers = br_memory_savers

        # For O_BR
        self._current_policy_tags_O_BR = None

        # For T_BR
        self._current_policy_tags_T_BR = None

        # For O_OPP
        self._current_policy_tags_O_AVG = None

    def init(self, sws_br, sws_avg, nfsp_iter):
        self._current_policy_tags_O_BR = np.empty(shape=self._t_prof.n_steps_br_per_iter_per_la, dtype=np.int32)
        self._current_policy_tags_T_BR = np.empty(shape=self._t_prof.n_steps_br_per_iter_per_la, dtype=np.int32)
        self._current_policy_tags_O_AVG = np.empty(shape=self._t_prof.n_steps_avg_per_iter_per_la, dtype=np.int32)
        for sw in sws_br:
            self._current_policy_tags_O_BR[sw.env_idx] = SeatActorBase.pick_training_policy(br_prob=self.sampler.antic)
            self._current_policy_tags_T_BR[sw.env_idx] = SeatActorBase.pick_training_policy(br_prob=self.sampler.antic)
            self._br_memory_savers[sw.env_idx].reset(range_idx=sw.range_idxs[self.owner])

        for sw in sws_avg:
            self._avg_memory_savers[sw.env_idx].reset(range_idx=sw.range_idxs[self.owner],
                                                      sample_weight=nfsp_iter if self._t_prof.linear else 1)
            self._current_policy_tags_O_AVG[sw.env_idx] = SeatActorBase.pick_training_policy(br_prob=self.sampler.antic)

    # -------------------------------------------------- BR --------------------------------------------------
    def update_if_terminal_for_br(self, step_wrappers, is_traverser=False):
        for sw in step_wrappers:
            if sw.TERMINAL:
                if is_traverser:
                    self._br_memory_savers[sw.env_idx].add_to_buf(
                        reward_p=sw.term_rew_all[self.owner],
                        terminal_obs=sw.term_obs,
                    )
                    self._br_memory_savers[sw.env_idx].reset(range_idx=sw.range_idxs[self.owner])
                    self._current_policy_tags_T_BR[sw.env_idx] = SeatActorBase.pick_training_policy(
                        br_prob=self.sampler.antic)
                else:
                    self._current_policy_tags_O_BR[sw.env_idx] = SeatActorBase.pick_training_policy(
                        br_prob=self.sampler.antic)

    def act_for_br_trav(self, step_wrappers):
        # Act
        SeatActorBase.act_mixed(step_wrappers=step_wrappers, br_learner=self.br_learner, owner=self.owner,
                                avg_learner=self.avg_learner, current_policy_tags=self._current_policy_tags_T_BR,
                                explore=True)

        # Add to memories
        for sw in step_wrappers:
            e_i = sw.env_idx
            self._br_memory_savers[e_i].add_experience(obs_t_before_acted=sw.obs,
                                                       a_selected_t=sw.action,
                                                       legal_actions_list_t=sw.legal_actions_list)

    def act_for_br_opp(self, step_wrappers):
        """ Anticipatory; greedy BR + AVG """
        SeatActorBase.act_mixed(step_wrappers=step_wrappers, br_learner=self.br_learner, owner=self.owner,
                                avg_learner=self.avg_learner, current_policy_tags=self._current_policy_tags_O_BR,
                                explore=True)

    # -------------------------------------------------- AVG --------------------------------------------------
    def update_if_terminal_for_avg(self, step_wrappers, nfsp_iter, is_traverser=False):
        for sw in step_wrappers:
            if sw.TERMINAL:
                if is_traverser:
                    self._avg_memory_savers[sw.env_idx].reset(range_idx=sw.range_idxs[self.owner],
                                                              sample_weight=nfsp_iter if self._t_prof.linear else 1
                                                              )
                else:
                    self._current_policy_tags_O_AVG[sw.env_idx] = SeatActorBase.pick_training_policy(
                        br_prob=self.sampler.antic)

    def reset_adv_trav(self, range_idx, weight):
        self._avg_memory_savers.reset(range_idx=range_idx, weight=weight)

    def act_for_avg_trav(self, step_wrappers):
        """ BR greedy """
        with torch.no_grad():
            if len(step_wrappers) > 0:
                actions, _ = SeatActorBase.choose_a_br(step_wrappers=step_wrappers, owner=self.owner,
                                                       br_learner=self.br_learner, random_prob=self.br_learner.eps)
                for a, sw in zip(actions, step_wrappers):
                    a = a.item()
                    sw.action = a
                    sw.action_was_random = False
                    self._avg_memory_savers[sw.env_idx].add_step(pub_obs=sw.obs,
                                                                 a=a,
                                                                 legal_actions_mask=rl_util.get_legal_action_mask_np(
                                                                     n_actions=self._env_bldr.N_ACTIONS,
                                                                     legal_actions_list=sw.legal_actions_list)
                                                                 )

    # TODO
    def act_for_avg_opp(self, step_wrappers):
        """
        Purely random because that's how it should be for correct reach
        """
        SeatActorBase.act_mixed(step_wrappers=step_wrappers, br_learner=self.br_learner, owner=self.owner,
                                avg_learner=self.avg_learner, current_policy_tags=self._current_policy_tags_O_AVG,
                                explore=True)

        # for sw in step_wrappers:
        #     sw.action = sw.legal_actions_list[np.random.randint(low=0, high=len(sw.legal_actions_list))]
