# Copyright (c) 2019 Eric Steinberger


import pickle

from PokerRL.rl import rl_util
from PokerRL.rl.base_cls.workers.ParameterServerBase import ParameterServerBase as _ParameterServerBase
from PokerRL.rl.neural import DuelingQNet
from PokerRL.rl.neural.AvrgStrategyNet import AvrgStrategyNet


class ParameterServer(_ParameterServerBase):

    def __init__(self, t_prof, seat_id, chief_handle):
        self.ddqn_args = t_prof.module_args["ddqn"]
        self.avg_args = t_prof.module_args["avg"]
        super().__init__(t_prof=t_prof, chief_handle=chief_handle)

        self.seat_id = seat_id
        self.global_iter_id = 0

        self.eps = self.ddqn_args.eps_start
        self.antic = self._t_prof.antic_start

        self.q_net = DuelingQNet(q_args=self.ddqn_args.q_args, env_bldr=self._env_bldr, device=self._device)
        self.avg_net = AvrgStrategyNet(avrg_net_args=self.avg_args.avg_net_args, env_bldr=self._env_bldr,
                                       device=self._device)

        self.br_optim = rl_util.str_to_optim_cls(self.ddqn_args.optim_str)(self.q_net.parameters(),
                                                                           lr=self.ddqn_args.lr)
        self.avg_optim = rl_util.str_to_optim_cls(self.avg_args.optim_str)(self.avg_net.parameters(),
                                                                           lr=self.avg_args.lr)

        self.eps_exp = self._ray.remote(self._chief_handle.create_experiment,
                                        t_prof.name + ": epsilon Plyr" + str(seat_id))
        self.antic_exp = self._ray.remote(self._chief_handle.create_experiment,
                                          t_prof.name + ": anticipatory Plyr" + str(seat_id))
        self._log_eps()
        self._log_antic()

    # ______________________________________________ API to pull from PS _______________________________________________
    def get_avg_weights(self):
        self.avg_net.zero_grad()
        return self._ray.state_dict_to_numpy(self.avg_net.state_dict())

    def get_q1_weights(self):
        self.q_net.zero_grad()
        return self._ray.state_dict_to_numpy(self.q_net.state_dict())

    def get_eps(self):
        return self.eps

    def get_antic(self):
        return self.antic

    def _log_eps(self):
        self._ray.remote(self._chief_handle.add_scalar,
                         self.eps_exp, "Epsilon", self.global_iter_id, self.eps)

    def _log_antic(self):
        self._ray.remote(self._chief_handle.add_scalar,
                         self.antic_exp, "Anticipatory Parameter", self.global_iter_id, self.antic)

    # ____________________________________________ API to make PS compute ______________________________________________
    def apply_grads_br(self, list_grads):
        self._apply_grads(list_of_grads=list_grads, optimizer=self.br_optim, net=self.q_net,
                          grad_norm_clip=self.ddqn_args.grad_norm_clipping)

    def apply_grads_avg(self, list_grads):
        self._apply_grads(list_of_grads=list_grads, optimizer=self.avg_optim, net=self.avg_net,
                          grad_norm_clip=self.avg_args.grad_norm_clipping)

    def increment(self):
        self.global_iter_id += 1

        self.eps = rl_util.polynomial_decay(base=self.ddqn_args.eps_start,
                                            const=self.ddqn_args.eps_const,
                                            exponent=self.ddqn_args.eps_exponent,
                                            minimum=self.ddqn_args.eps_min,
                                            counter=self.global_iter_id)
        self.antic = rl_util.polynomial_decay(base=self._t_prof.antic_start,
                                              const=self._t_prof.antic_const,
                                              exponent=self._t_prof.antic_exponent,
                                              minimum=self._t_prof.antic_min,
                                              counter=self.global_iter_id)

        if self.global_iter_id % 1000 == 0:
            self._log_eps()
            self._log_antic()
        return self.seat_id

    # ______________________________________________ API for checkpointing _____________________________________________
    def checkpoint(self, curr_step):
        state = {
            "seat_id": self.seat_id,
            "eps": self.eps,
            "antic": self.antic,
            "q_net": self.q_net.state_dict(),
            "avg_net": self.avg_net.state_dict(),
            "br_optim": self.br_optim.state_dict(),
            "avg_optim": self.avg_optim.state_dict(),
        }

        with open(self._get_checkpoint_file_path(name=self._t_prof.name, step=curr_step,
                                                 cls=self.__class__, worker_id="P" + str(self.seat_id)),
                  "wb") as pkl_file:
            pickle.dump(obj=state, file=pkl_file, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, name_to_load, step):
        with open(self._get_checkpoint_file_path(name=name_to_load, step=step,
                                                 cls=self.__class__, worker_id="P" + str(self.seat_id)),
                  "rb") as pkl_file:
            state = pickle.load(pkl_file)

            assert self.seat_id == state["seat_id"]

            self.eps = state["eps"]
            self.antic = state["antic"]
            self.q_net.load_state_dict(state["q_net"])
            self.avg_net.load_state_dict(state["avg_net"])
            self.br_optim.load_state_dict(state["br_optim"])
            self.avg_optim.load_state_dict(state["avg_optim"])
