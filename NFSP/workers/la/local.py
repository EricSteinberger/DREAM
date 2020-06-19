# Copyright (c) 2019 Eric Steinberger


import pickle

from NFSP.AvgWrapper import AvgWrapper
from NFSP.workers.la.playing.AdamSampler import AdamSampler
from NFSP.workers.la.playing.CleanSampler import CleanSampler
from NFSP.workers.la.playing.VanillaSampler import VanillaSampler
from PokerRL.rl import rl_util
from PokerRL.rl.agent_modules import DDQN
from PokerRL.rl.base_cls.workers.WorkerBase import WorkerBase


class LearnerActor(WorkerBase):
    """
    Methods for acting are not included in this base.
    """

    def __init__(self, t_prof, worker_id, chief_handle):
        super().__init__(t_prof=t_prof)

        self._env_bldr = rl_util.get_env_builder(t_prof=t_prof)
        self._id = worker_id
        self._chief_handle = chief_handle

        self._ddqn_args = t_prof.module_args["ddqn"]
        self._avg_args = t_prof.module_args["avg"]

        if t_prof.nn_type == "recurrent":
            from PokerRL.rl.buffers.CircularBufferRNN import CircularBufferRNN
            from NFSP.workers.la.action_buffer.ActionBufferRNN import ActionBufferRNN

            BR_BUF_CLS = CircularBufferRNN
            AVG_BUF_CLS = ActionBufferRNN

        elif t_prof.nn_type == "feedforward":
            from PokerRL.rl.buffers.CircularBufferFLAT import CircularBufferFLAT
            from NFSP.workers.la.action_buffer.ActionBufferFLAT import ActionBufferFLAT

            BR_BUF_CLS = CircularBufferFLAT
            AVG_BUF_CLS = ActionBufferFLAT
        else:
            raise ValueError(t_prof.nn_type)

        self._avg_buf2 = [
            AVG_BUF_CLS(env_bldr=self._env_bldr, max_size=self._avg_args.res_buf_size,
                        min_prob=self._avg_args.min_prob_res_buf)
            for p in range(self._env_bldr.N_SEATS)
        ]
        self._br_buf2 = [
            BR_BUF_CLS(env_bldr=self._env_bldr, max_size=self._ddqn_args.cir_buf_size)
            for p in range(self._env_bldr.N_SEATS)
        ]
        self._br_learner2 = [
            DDQN(owner=p, ddqn_args=self._ddqn_args, env_bldr=self._env_bldr)
            for p in range(self._env_bldr.N_SEATS)
        ]
        self._avg_learner2 = [
            AvgWrapper(owner=p, env_bldr=self._env_bldr, avg_training_args=self._avg_args)
            for p in range(self._env_bldr.N_SEATS)
        ]

        if self._t_prof.sampling == "adam":
            self._sampler = AdamSampler(t_prof=t_prof, env_bldr=self._env_bldr, br_buf2=self._br_buf2,
                                        avg_buf2=self._avg_buf2, br_learner2=self._br_learner2,
                                        avg_learner2=self._avg_learner2, constant_eps=self._t_prof.constant_eps_expl)

        elif self._t_prof.sampling == "clean":
            self._sampler = CleanSampler(t_prof=t_prof, env_bldr=self._env_bldr, br_buf2=self._br_buf2,
                                         avg_buf2=self._avg_buf2, br_learner2=self._br_learner2,
                                         avg_learner2=self._avg_learner2, constant_eps=self._t_prof.constant_eps_expl)
        else:
            self._sampler = VanillaSampler(t_prof=t_prof, env_bldr=self._env_bldr, br_buf2=self._br_buf2,
                                           avg_buf2=self._avg_buf2, br_learner2=self._br_learner2,
                                           avg_learner2=self._avg_learner2)

    # ____________________________________________________ Playing _____________________________________________________
    def play(self, nfsp_iter):
        self._all_eval()
        return self._sampler.play(nfsp_iter=nfsp_iter)

    # ____________________________________________________ Learning ____________________________________________________
    def get_br_grads(self, p_id):
        self._br_learner2[p_id].train()
        g = self._br_learner2[p_id].get_grads_one_batch_from_buffer(buffer=self._br_buf2[p_id])
        if g is None:
            return None
        return self._ray.grads_to_numpy(g)

    def get_avg_grads(self, p_id):
        self._avg_learner2[p_id].train()
        g = self._avg_learner2[p_id].get_grads_one_batch_from_buffer(buffer=self._avg_buf2[p_id])
        if g is None:
            return None
        return self._ray.grads_to_numpy(g)

    def update(self,
               p_id,
               q1_state_dict,
               avg_state_dict,
               eps,
               antic,
               ):
        if q1_state_dict is not None:
            dict_torch = self._ray.state_dict_to_torch(q1_state_dict, device=self._br_learner2[p_id].device)
            self._br_learner2[p_id].load_net_state_dict(dict_torch)

        if avg_state_dict is not None:
            dict_torch = self._ray.state_dict_to_torch(avg_state_dict, device=self._avg_learner2[p_id].device)
            self._avg_learner2[p_id].load_net_state_dict(dict_torch)

        if eps is not None:
            self._br_learner2[p_id].eps = eps

        if eps is not None:
            self._sampler.antic = antic

    def update_q2(self, p_id):
        self._br_learner2[p_id].update_target_net()

    def empty_cir_bufs(self):
        for b in self._br_buf2:
            b.reset()

    # __________________________________________________________________________________________________________________
    def checkpoint(self, curr_step):
        for p_id in range(self._env_bldr.N_SEATS):
            state = {
                "pi": self._avg_learner2[p_id].state_dict(),
                "br": self._br_learner2[p_id].state_dict(),
                "cir": self._br_buf2[p_id].state_dict(),
                "res": self._avg_buf2[p_id].state_dict(),
                "p_id": p_id,
            }
            with open(self._get_checkpoint_file_path(name=self._t_prof.name, step=curr_step,
                                                     cls=self.__class__, worker_id=str(self._id) + "_P" + str(p_id)),
                      "wb") as pkl_file:
                pickle.dump(obj=state, file=pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self._get_checkpoint_file_path(name=self._t_prof.name, step=curr_step,
                                                 cls=self.__class__, worker_id=str(self._id) + "_General"),
                  "wb") as pkl_file:
            state = {
                "env": self._parallel_env.state_dict()
            }
            pickle.dump(obj=state, file=pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, name_to_load, step):
        for p_id in range(self._env_bldr.N_SEATS):
            with open(self._get_checkpoint_file_path(name=name_to_load, step=step,
                                                     cls=self.__class__, worker_id=str(self._id) + "_P" + str(p_id)),
                      "rb") as pkl_file:
                state = pickle.load(pkl_file)

                assert state["p_id"] == p_id

                self._avg_learner2[p_id].load_state_dict(state["avg"])
                self._br_learner2[p_id].load_state_dict(state["br"])
                self._br_buf2[p_id].load_state_dict(state["cir"])
                self._avg_buf2[p_id].load_state_dict(state["res"])

        with open(self._get_checkpoint_file_path(name=name_to_load, step=step,
                                                 cls=self.__class__, worker_id=str(self._id) + "_General"),
                  "rb") as pkl_file:
            state = pickle.load(pkl_file)
            self._parallel_env.load_state_dict(state["env"])
            self._last_step_wrappers = self._parallel_env.reset()

    def _all_eval(self):
        for q in self._br_learner2:
            q.eval()
        for a_l in self._avg_learner2:
            a_l.eval()

    def _all_train(self):
        for q in self._br_learner2:
            q.train()
        for a_l in self._avg_learner2:
            a_l.train()
