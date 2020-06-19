# Copyright (c) Eric Steinberger 2020

import os

import psutil

from DREAM_and_DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DREAM_and_DeepCFR.IterationStrategy import IterationStrategy
from DREAM_and_DeepCFR.workers.la.AdvWrapper import AdvWrapper
from DREAM_and_DeepCFR.workers.la.AvrgWrapper import AvrgWrapper
from DREAM_and_DeepCFR.workers.la.buffers.AdvReservoirBuffer import AdvReservoirBuffer
from DREAM_and_DeepCFR.workers.la.buffers.AvrgReservoirBuffer import AvrgReservoirBuffer
from DREAM_and_DeepCFR.workers.la.buffers.CrazyBaselineQCircularBuffer import CrazyBaselineQCircularBuffer
from DREAM_and_DeepCFR.workers.la.sampling_algorithms.ExternalSampler import ExternalSampler
from DREAM_and_DeepCFR.workers.la.sampling_algorithms.LearnedBaselineLearner import BaselineWrapper
from DREAM_and_DeepCFR.workers.la.sampling_algorithms.LearnedBaselineSampler import LearnedBaselineSampler
from DREAM_and_DeepCFR.workers.la.sampling_algorithms.MultiOutcomeSampler import MultiOutcomeSampler
from PokerRL.rl import rl_util
from PokerRL.rl.base_cls.workers.WorkerBase import WorkerBase


class LearnerActor(WorkerBase):

    def __init__(self, t_prof, worker_id, chief_handle):
        super().__init__(t_prof=t_prof)

        self._adv_args = t_prof.module_args["adv_training"]

        self._env_bldr = rl_util.get_env_builder(t_prof=t_prof)
        self._id = worker_id
        self._chief_handle = chief_handle

        self._adv_buffers = [
            AdvReservoirBuffer(owner=p, env_bldr=self._env_bldr, max_size=self._adv_args.max_buffer_size,
                               nn_type=t_prof.nn_type,
                               iter_weighting_exponent=self._t_prof.iter_weighting_exponent)
            for p in range(self._t_prof.n_seats)
        ]

        self._adv_wrappers = [
            AdvWrapper(owner=p,
                       env_bldr=self._env_bldr,
                       adv_training_args=self._adv_args,
                       device=self._adv_args.device_training)
            for p in range(self._t_prof.n_seats)
        ]

        self._AVRG = EvalAgentDeepCFR.EVAL_MODE_AVRG_NET in self._t_prof.eval_modes_of_algo
        self._SINGLE = EvalAgentDeepCFR.EVAL_MODE_SINGLE in self._t_prof.eval_modes_of_algo

        # """"""""""""""""""""""""""""
        # Deep CFR
        # """"""""""""""""""""""""""""
        if self._AVRG:
            self._avrg_args = t_prof.module_args["avrg_training"]

            self._avrg_buffers = [
                AvrgReservoirBuffer(owner=p, env_bldr=self._env_bldr, max_size=self._avrg_args.max_buffer_size,
                                    nn_type=t_prof.nn_type,
                                    iter_weighting_exponent=self._t_prof.iter_weighting_exponent)
                for p in range(self._t_prof.n_seats)
            ]

            self._avrg_wrappers = [
                AvrgWrapper(owner=p,
                            env_bldr=self._env_bldr,
                            avrg_training_args=self._avrg_args,
                            device=self._avrg_args.device_training)
                for p in range(self._t_prof.n_seats)
            ]

            if self._t_prof.sampler.lower() == "mo":
                self._data_sampler = MultiOutcomeSampler(
                    env_bldr=self._env_bldr,
                    adv_buffers=self._adv_buffers,
                    avrg_buffers=self._avrg_buffers,
                    n_actions_traverser_samples=self._t_prof.n_actions_traverser_samples)

            elif self._t_prof.sampler.lower() == "learned_baseline":
                assert t_prof.module_args["mccfr_baseline"] is not None, "Please give 'baseline_args' for VR Sampler."
                self._baseline_args = t_prof.module_args["mccfr_baseline"]
                self._baseline_wrapper = BaselineWrapper(env_bldr=self._env_bldr,
                                                         baseline_args=self._baseline_args)

                self._baseline_buf = CrazyBaselineQCircularBuffer(owner=None, env_bldr=self._env_bldr,
                                                                  max_size=self._baseline_args.max_buffer_size,
                                                                  nn_type=t_prof.nn_type)

                self._data_sampler = LearnedBaselineSampler(
                    env_bldr=self._env_bldr, adv_buffers=self._adv_buffers, eps=self._t_prof.os_eps,
                    baseline_net=self._baseline_wrapper, baseline_buf=self._baseline_buf,
                    avrg_buffers=self._avrg_buffers,
                )
            else:
                raise ValueError("Currently we don't support", self._t_prof.sampler.lower(), "sampling.")
        else:
            if self._t_prof.sampler.lower() == "learned_baseline":
                assert t_prof.module_args["mccfr_baseline"] is not None, "Please give 'baseline_args' for VR Sampler."
                self._baseline_args = t_prof.module_args["mccfr_baseline"]
                self._baseline_wrapper = BaselineWrapper(env_bldr=self._env_bldr,
                                                         baseline_args=self._baseline_args)

                self._baseline_buf = CrazyBaselineQCircularBuffer(owner=None, env_bldr=self._env_bldr,
                                                                  max_size=self._baseline_args.max_buffer_size,
                                                                  nn_type=t_prof.nn_type)

                self._data_sampler = LearnedBaselineSampler(
                    env_bldr=self._env_bldr, adv_buffers=self._adv_buffers, eps=self._t_prof.os_eps,
                    baseline_net=self._baseline_wrapper, baseline_buf=self._baseline_buf,
                )

            elif self._t_prof.sampler.lower() == "es":
                self._data_sampler = ExternalSampler(
                    env_bldr=self._env_bldr,
                    adv_buffers=self._adv_buffers,
                    avrg_buffers=None,
                )

            elif self._t_prof.sampler.lower() == "mo":
                self._data_sampler = MultiOutcomeSampler(
                    env_bldr=self._env_bldr,
                    adv_buffers=self._adv_buffers,
                    avrg_buffers=None,
                    eps=self._t_prof.os_eps,
                    n_actions_traverser_samples=self._t_prof.n_actions_traverser_samples)
            else:
                raise ValueError("Currently we don't support", self._t_prof.sampler.lower(), "sampling.")

        if self._t_prof.log_verbose:
            self._exp_mem_usage = self._ray.get(
                self._ray.remote(self._chief_handle.create_experiment,
                                 self._t_prof.name + "_LA" + str(worker_id) + "_Memory_Usage"))
            self._exps_adv_buffer_size = self._ray.get(
                [
                    self._ray.remote(self._chief_handle.create_experiment,
                                     self._t_prof.name + "_LA" + str(worker_id) + "_P" + str(p) + "_ADV_BufSize")
                    for p in range(self._t_prof.n_seats)
                ]
            )
            if self._AVRG:
                self._exps_avrg_buffer_size = self._ray.get(
                    [
                        self._ray.remote(self._chief_handle.create_experiment,
                                         self._t_prof.name + "_LA" + str(worker_id) + "_P" + str(p) + "_AVRG_BufSize")
                        for p in range(self._t_prof.n_seats)
                    ]
                )

    def generate_data(self, traverser, cfr_iter):
        iteration_strats = [
            IterationStrategy(t_prof=self._t_prof, env_bldr=self._env_bldr, owner=p,
                              device=self._t_prof.device_inference, cfr_iter=cfr_iter)
            for p in range(self._t_prof.n_seats)
        ]
        for s in iteration_strats:
            s.load_net_state_dict(state_dict=self._adv_wrappers[s.owner].net_state_dict())

        self._data_sampler.generate(n_traversals=self._t_prof.n_traversals_per_iter,
                                    traverser=traverser,
                                    iteration_strats=iteration_strats,
                                    cfr_iter=cfr_iter,
                                    )

        # Log after both players generated data
        if self._t_prof.log_verbose and traverser == 1 and (cfr_iter % 3 == 0):
            for p in range(self._t_prof.n_seats):
                self._ray.remote(self._chief_handle.add_scalar,
                                 self._exps_adv_buffer_size[p], "Debug/BufferSize", cfr_iter,
                                 self._adv_buffers[p].size)
                if self._AVRG:
                    self._ray.remote(self._chief_handle.add_scalar,
                                     self._exps_avrg_buffer_size[p], "Debug/BufferSize", cfr_iter,
                                     self._avrg_buffers[p].size)

            process = psutil.Process(os.getpid())
            self._ray.remote(self._chief_handle.add_scalar,
                             self._exp_mem_usage, "Debug/MemoryUsage/LA", cfr_iter,
                             process.memory_info().rss)

        return self._data_sampler.total_node_count_traversed

    def update(self, adv_state_dicts=None, avrg_state_dicts=None, baseline_state_dict=None):
        """
        Args:
            adv_state_dicts (list):         Optional. if not None:
                                                        expects a list of neural net state dicts or None for each player
                                                        in order of their seat_ids. This allows updating only some
                                                        players.

            avrg_state_dicts (list):         Optional. if not None:
                                                        expects a list of neural net state dicts or None for each player
                                                        in order of their seat_ids. This allows updating only some
                                                        players.
        """
        baseline_state_dict = baseline_state_dict[0]  # wrapped bc of object id stuff
        if baseline_state_dict is not None:
            self._baseline_wrapper.load_net_state_dict(
                state_dict=self._ray.state_dict_to_torch(self._ray.get(baseline_state_dict),
                                                         device=self._baseline_wrapper.device))

        for p_id in range(self._t_prof.n_seats):
            if adv_state_dicts[p_id] is not None:
                self._adv_wrappers[p_id].load_net_state_dict(
                    state_dict=self._ray.state_dict_to_torch(self._ray.get(adv_state_dicts[p_id]),
                                                             device=self._adv_wrappers[p_id].device))

            if avrg_state_dicts[p_id] is not None:
                self._avrg_wrappers[p_id].load_net_state_dict(
                    state_dict=self._ray.state_dict_to_torch(self._ray.get(avrg_state_dicts[p_id]),
                                                             device=self._avrg_wrappers[p_id].device))

    def get_loss_last_batch_adv(self, p_id):
        return self._adv_wrappers[p_id].loss_last_batch

    def get_loss_last_batch_avrg(self, p_id):
        return self._avrg_wrappers[p_id].loss_last_batch

    def get_loss_last_batch_baseline(self):
        return self._baseline_wrapper.loss_last_batch

    def get_adv_grads(self, p_id):
        return self._ray.grads_to_numpy(
            self._adv_wrappers[p_id].get_grads_one_batch_from_buffer(buffer=self._adv_buffers[p_id]))

    def get_avrg_grads(self, p_id):
        return self._ray.grads_to_numpy(
            self._avrg_wrappers[p_id].get_grads_one_batch_from_buffer(buffer=self._avrg_buffers[p_id]))

    def get_baseline_grads(self):
        return self._ray.grads_to_numpy(
            self._baseline_wrapper.get_grads_one_batch_from_buffer(buffer=self._baseline_buf))
