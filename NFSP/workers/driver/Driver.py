# Copyright (c) 2019 Eric Steinberger


import pickle

from NFSP.EvalAgentNFSP import EvalAgentNFSP
from NFSP.workers.driver._HighLevelAlgo import HighLevelAlgo
from PokerRL.rl.base_cls.workers.DriverBase import DriverBase as _DriverBase


class Driver(_DriverBase):

    def __init__(self,
                 t_prof, eval_methods, n_iterations=None, iteration_to_import=None, name_to_import=None):
        if t_prof.DISTRIBUTED:
            from NFSP.workers.chief.dist import Chief
            from NFSP.workers.la.dist import LearnerActor
            from NFSP.workers.ps.dist import ParameterServer

        else:
            from NFSP.workers.chief.local import Chief
            from NFSP.workers.la.local import LearnerActor
            from NFSP.workers.ps.local import ParameterServer
        # __________________________________________ Create and init workers  __________________________________________
        super().__init__(t_prof=t_prof, eval_methods=eval_methods, n_iterations=n_iterations,
                         iteration_to_import=iteration_to_import, name_to_import=name_to_import,
                         chief_cls=Chief, eval_agent_cls=EvalAgentNFSP)

        print("Creating LAs...")
        self.la_handles = [
            self._ray.create_worker(LearnerActor,
                                    t_prof,
                                    i,
                                    self.chief_handle)
            for i in range(t_prof.n_learner_actors)
        ]

        print("Creating Parameter Servers...")
        self.ps_handles = [self._ray.create_worker(ParameterServer,
                                                   t_prof,
                                                   p_id,
                                                   self.chief_handle)
                           for p_id in range(t_prof.n_seats)
                           ]

        self._ray.wait([self._ray.remote(self.chief_handle.set_ps_handles,
                                         *self.ps_handles),
                        self._ray.remote(self.chief_handle.set_la_handles,
                                         *self.la_handles  # This is not supported otherwise in ray.. :(
                                         )
                        ])

        print("Created and initialized Workers")

        self.algo = HighLevelAlgo(t_prof=t_prof,
                                  la_handles=self.la_handles,
                                  ps_handles=self.ps_handles,
                                  chief_handle=self.chief_handle)

        self._maybe_load_checkpoint_init()

    def run(self):
        print("Setting stuff up...")
        self.algo.init()

        for _iter_nr in range(10000000 if self.n_iterations is None else self.n_iterations):

            # Evaluate. Sync & Lock, then train while evaluating on other workers
            self.evaluate()

            if self._iteration % self._t_prof.log_export_freq == 0:
                self.save_logs()

            print("Iteration: ", self._iteration)

            # _____________________________________________ play ___________________________________________________
            if self._iteration == 0:
                times = self.algo.run_one_iter(
                    n_avg_updates=0,
                    n_br_updates=self._t_prof.n_br_updates_per_iter * self._t_prof.training_multiplier_iter_0,
                    nfsp_iter=_iter_nr,
                )
            else:
                times = self.algo.run_one_iter(
                    n_avg_updates=self._t_prof.n_avg_updates_per_iter,
                    n_br_updates=self._t_prof.n_br_updates_per_iter,
                    nfsp_iter=_iter_nr,
                )

            self._iteration += 1

            self.periodically_export_eval_agent()
            self.periodically_checkpoint()

            print("Played ", times["t_playing"], "s.",
                  "  ||  Trained", times["t_computation"], " s.",
                  "  ||  Syncing took", times["t_syncing"], " s.",
                  )

    def load_checkpoint(self, name_to_load, step):
        print("loading from iteration: ", step)

        # Load Checkpoint of Driver
        with open(self._get_checkpoint_file_path(name=name_to_load, step=step,
                                                 cls=self.__class__, worker_id=""),
                  "rb") as pkl_file:
            self.algo.load_state_dict(pickle.load(pkl_file))

        # Call on all other workers sequentially to be safe against RAM overload
        for w in self.la_handles + self.ps_handles:
            self._ray.wait([
                self._ray.remote(w.load_checkpoint,
                                 name_to_load, step)
            ])

    def checkpoint(self, **kwargs):
        """
        store state of the whole system to be able to stop now and resume training later
        pickles ALL le_act workers and ALL t_profervers and saves that to Storage Server.
        """

        # Load Checkpoint of Driver
        with open(self._get_checkpoint_file_path(name=self._t_prof.name, step=self._iteration,
                                                 cls=self.__class__, worker_id=""),
                  "wb") as pkl_file:
            pickle.dump(obj=self.algo.state_dict(), file=pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

        # Call on all other workers sequentially to be safe against RAM overload
        for w in self.la_handles + self.ps_handles:
            self._ray.wait([
                self._ray.remote(w.checkpoint,
                                 self._iteration)
            ])

        # Delete past checkpoints
        s = [self._iteration]
        if self._iteration > self._t_prof.checkpoint_freq + 1:
            s.append(self._iteration - self._t_prof.checkpoint_freq)

        self._delete_past_checkpoints(steps_not_to_delete=s)
