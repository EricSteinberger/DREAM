# Copyright (c) 2019 Eric Steinberger


import copy

import numpy as np

from PokerRL.eval._.EvaluatorMasterBase import EvaluatorMasterBase
from PokerRL.eval.rl_br import _util


class LocalRLBRMaster(EvaluatorMasterBase):

    def __init__(self, t_prof, chief_handle, eval_agent_cls):
        super().__init__(t_prof=t_prof,
                         eval_env_bldr=_util.get_env_builder_rlbr(t_prof=t_prof),
                         chief_handle=chief_handle,
                         evaluator_name="RL-BR",
                         log_conf_interval=True)

        assert self._eval_env_bldr.N_SEATS == 2, "only works for 2 players at the moment"

        self._args = t_prof.module_args["rlbr"]
        self._eval_agent = eval_agent_cls(t_prof=t_prof)

        self._la_handles_0 = None
        self._la_handles_1 = None
        self._ps_handle = None

        if self._t_prof.log_verbose and self._args.n_brs_to_train > 1:
            self._exps_br_quality = {
                eval_mode:
                    [
                        [
                            self._ray.get(self._ray.remote(self._chief_handle.create_experiment,
                                                           self._t_prof.name
                                                           + " " + eval_mode
                                                           + "_stack_" + str(stack_size[0])
                                                           + ": " + self._evaluator_name
                                                           + " RLBR Quality"
                                                           + "_brID" + str(_br_id)
                                                           ))
                            for _br_id in range(self._args.n_brs_to_train)
                        ]
                        for stack_size in self._t_prof.eval_stack_sizes
                    ]
                for eval_mode in self._t_prof.eval_modes_of_algo
            }

    def set_learner_actors_0(self, *las):
        self._la_handles_0 = list(las)

    def set_learner_actors_1(self, *las):
        self._la_handles_1 = list(las)

    def set_param_server(self, param_server):
        self._ps_handle = param_server

    def update_weights(self):
        w = self.pull_current_strat_from_chief()
        self._eval_agent.update_weights(copy.deepcopy(w))

    def evaluate(self, global_iter_nr):
        for mode in self._t_prof.eval_modes_of_algo:
            for stack_size_idx, stack_size in enumerate(self._t_prof.eval_stack_sizes):
                self._eval_agent.set_mode(mode=mode)
                self._eval_agent.set_stack_size(stack_size=stack_size)
                if self._eval_agent.can_compute_mode():

                    if self._args.n_brs_to_train == 1:  # Calculate conf interval based on individual game outcomes
                        scores, _ = self._retrain_and_eval(br_number=0, mode=mode, stack_size=stack_size,
                                                           stack_size_idx=stack_size_idx,
                                                           global_iter_nr=global_iter_nr)

                    else:  # Train multiple BRs and use the best to avoid the typical RL variance.
                        best_score = None
                        best_br_state_dicts = None
                        all_brs_mean_rew = []
                        for br_id in range(self._args.n_brs_to_train):
                            _scores_from_this_br, _br_state_dicts = \
                                self._retrain_and_eval(br_number=br_id, mode=mode, stack_size=stack_size,
                                                       stack_size_idx=stack_size_idx, global_iter_nr=global_iter_nr)

                            s = sum(_scores_from_this_br) / len(_scores_from_this_br)
                            all_brs_mean_rew.append(s)
                            if best_score is None or s > best_score:
                                best_score = s
                                best_br_state_dicts = _br_state_dicts

                        for br_rank, score in enumerate(sorted(all_brs_mean_rew)):
                            self._ray.remote(self._chief_handle.add_scalar,
                                             self._exps_br_quality[mode][stack_size_idx][br_rank], "RL-BR/BR Quality",
                                             global_iter_nr, score)

                        # Now rerun eval for the best BR
                        scores = self._compute_rlbr(stack_size=stack_size, )

                    mean, d = self._get_95confidence(scores=scores)

                    self._log_results(iter_nr=global_iter_nr, agent_mode=mode, stack_size_idx=stack_size_idx,
                                      score=mean, lower_conf95=mean - d, upper_conf95=mean + d)

    def _retrain_and_eval(self, br_number, mode, stack_size, stack_size_idx, global_iter_nr):
        self._retrain(mode=mode, stack_size=stack_size, stack_size_idx=stack_size_idx,
                      global_iter_nr=global_iter_nr, br_number=br_number)
        # """""""""""""""""""
        # Compute RL-BR
        # """""""""""""""""""
        print("Running rollout matches between RL-BR and agent.")
        ddqn_states = self._ray.get(self._ray.remote(self._ps_handle.get_eval_ddqn_state_dicts))

        scores = self._compute_rlbr(stack_size=stack_size)

        return scores, ddqn_states

    def _retrain(self, br_number, mode, stack_size, stack_size_idx, global_iter_nr):
        ALL_PLYRS = [0, 1]
        P_LA_ZIPPED = list(
            zip([0 for _ in range(len(self._la_handles_0))] + [1 for _ in range(len(self._la_handles_1))],
                self._la_handles_0 + self._la_handles_1))

        # """""""""""""""""""
        # Prepare Logging
        # """""""""""""""""""
        running_rew_exp = self._ray.get(
            self._ray.remote(self._chief_handle.create_experiment,
                             self._t_prof.name + "_M_" + mode + "_S" + str(stack_size_idx) + "_I" + str(
                                 global_iter_nr) + "Running Rew RL-BR __" + str(br_number)))
        if self._t_prof.log_verbose:
            eps_exp = self._ray.get([
                self._ray.remote(self._chief_handle.create_experiment,
                                 self._t_prof.name + "_M_" + mode + "_S" + str(stack_size_idx) + "_I" + str(
                                     global_iter_nr) + "Epsilon RL-BR__" + str(br_number))
            ])

        logging_scores = []
        logging_eps = []
        logging_timesteps = []

        # """""""""""""""""""
        # Training
        # """""""""""""""""""

        print("Training RL-BR with agent mode", mode, "and stack size", stack_size_idx)

        # """"""""
        # Reset
        # """"""""
        self._eval_agent.reset()
        self._ray.wait([
            self._ray.remote(la.reset,
                             p, self._eval_agent.state_dict(), stack_size)
            for p, la in P_LA_ZIPPED
        ])
        self._ray.wait([
            self._ray.remote(self._ps_handle.reset, p, global_iter_nr)
            for p in range(self._eval_env_bldr.N_SEATS)
        ])
        self._update_leaner_actors(update_eps_for_plyrs=ALL_PLYRS, update_net_for_plyrs=ALL_PLYRS)

        # """"""""
        # Pre-play
        # """"""""
        self._ray.wait([
            self._ray.remote(la.play, self._args.pretrain_n_steps)
            for la in self._la_handles_0 + self._la_handles_1
        ])

        # """"""""
        # Learn
        # """"""""
        SMOOTHING = 200
        accum_score = 0.0
        for training_iter_id in range(self._args.n_iterations):
            for p in range(self._eval_env_bldr.N_SEATS):
                self._ray.wait([
                    self._ray.remote(self._ps_handle.update_eps, p, training_iter_id)
                ])
            self._update_leaner_actors(update_eps_for_plyrs=ALL_PLYRS)

            # Play
            scores_all_las = self._ray.get([
                self._ray.remote(la.play, self._args.play_n_steps_per_iter_per_la)
                for la in self._la_handles_0 + self._la_handles_1
            ])

            accum_score += sum(scores_all_las) / (len(self._la_handles_0) + len(self._la_handles_1))

            # Get Gradients
            grads_from_all_las_0, grads_from_all_las_1 = self._get_gradients()

            # Applying gradients
            self._ray.wait([
                self._ray.remote(self._ps_handle.apply_grads,
                                 0, grads_from_all_las_0),
                self._ray.remote(self._ps_handle.apply_grads,
                                 1, grads_from_all_las_1),
            ])

            # Update weights on all las
            self._update_leaner_actors(update_net_for_plyrs=ALL_PLYRS)

            # Periodically update target net
            if (training_iter_id + 1) % self._args.ddqn_args.target_net_update_freq:
                self._ray.wait([
                    self._ray.remote(la.update_target_net,
                                     p)
                    for p, la in P_LA_ZIPPED
                ])

            if (training_iter_id + 1) % SMOOTHING == 0:
                print("RL-BR iter", training_iter_id + 1)
                accum_score /= SMOOTHING
                logging_scores.append(accum_score)
                logging_eps.append(
                    self._ray.get(self._ray.remote(self._ps_handle.get_eps, 0)))
                logging_timesteps.append(training_iter_id + 1)
                accum_score = 0.0

        # """""""""""""""""""
        # Logging
        # """""""""""""""""""
        for i, logging_iter in enumerate(logging_timesteps):
            self._ray.remote(
                self._chief_handle.add_scalar,
                running_rew_exp, "RL-BR/Running Reward While Training", logging_iter,
                logging_scores[i])

        if self._t_prof.log_verbose:
            for i, logging_iter in enumerate(logging_timesteps):
                self._ray.remote(
                    self._chief_handle.add_scalar,
                    eps_exp, "RL-BR/Training Epsilon", logging_iter, logging_eps[i])

    def _get_gradients(self):
        grads_0 = [
            self._ray.remote(la.get_grads,
                             0)
            for la in self._la_handles_0
        ]
        grads_1 = [
            self._ray.remote(la.get_grads,
                             1)
            for la in self._la_handles_1
        ]
        g = self._ray.wait(grads_0 + grads_1)
        return g[0:len(grads_0)], g[len(grads_0):len(grads_0) + len(grads_1)]

    def _update_leaner_actors(self, update_eps_for_plyrs=None, update_net_for_plyrs=None):
        assert isinstance(update_net_for_plyrs, list) or update_net_for_plyrs is None
        assert isinstance(update_eps_for_plyrs, list) or update_eps_for_plyrs is None

        _update_net_per_p = [
            True if (update_net_for_plyrs is not None) and (p in update_net_for_plyrs) else False
            for p in range(self._t_prof.n_seats)
        ]

        _update_eps_per_p = [
            True if (update_eps_for_plyrs is not None) and (p in update_eps_for_plyrs) else False
            for p in range(self._t_prof.n_seats)
        ]

        for p_id, las in zip(list(range(self._t_prof.n_seats)), [self._la_handles_0, self._la_handles_1]):
            eps = [None for _ in range(self._t_prof.n_seats)]
            nets = [None for _ in range(self._t_prof.n_seats)]
            eps[p_id] = None if not _update_eps_per_p[p_id] else self._ray.remote(
                self._ps_handle.get_eps, p_id)

            nets[p_id] = None if not _update_net_per_p[p_id] else self._ray.remote(
                self._ps_handle.get_weights, p_id)
            self._ray.wait([
                self._ray.remote(la.update,
                                 eps,
                                 nets)
                for la in las
            ])

    def _compute_rlbr(self, stack_size):
        ddqn_states = self._ray.remote(self._ps_handle.get_eval_ddqn_state_dicts)
        outcomes = self._ray.get([
            self._ray.remote(la.compute_rlbr,
                             self._args.n_hands_each_seat_per_la,
                             ddqn_states,
                             stack_size
                             )
            for la in self._la_handles_0 + self._la_handles_1
        ])

        return np.array([x for la_outcomes in outcomes for x in la_outcomes])
