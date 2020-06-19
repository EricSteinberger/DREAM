# Copyright (c) 2019 Eric Steinberger


from os.path import join as ospj

from NFSP.EvalAgentNFSP import EvalAgentNFSP
from PokerRL.rl import rl_util
from PokerRL.rl.base_cls.workers.ChiefBase import ChiefBase
from PokerRL.util import file_util


class Chief(ChiefBase):

    def __init__(self, t_prof):
        super().__init__(t_prof=t_prof)
        self._t_prof = t_prof
        self._env_bldr = rl_util.get_env_builder(t_prof=t_prof)

        self._ps_handles = None
        self._la_handles = None

    def set_la_handles(self, *la_handles):
        self._la_handles = list(la_handles)

    def set_ps_handles(self, *ps_handles):
        self._ps_handles = list(ps_handles)

    def update_alive_las(self, alive_la_handles):
        self._la_handles = alive_la_handles

    # ________________________________________ Add and current models from PS __________________________________________
    def pull_current_eval_strategy(self, last_iteration_receiver_has=None):
        """ Pulls the newest Avg Net (obj ids if ray) from the PSs and sends them on. """
        _l = [
            self._ray.get(self._ray.remote(ps.get_avg_weights))
            for ps in self._ps_handles
        ]
        return _l

    # ________________________________ Store a pickled API class to play against the AI ________________________________
    def export_agent(self, step):
        _dir = ospj(self._t_prof.path_agent_export_storage, str(self._t_prof.name), str(step))
        file_util.create_dir_if_not_exist(_dir)

        eval_agent = EvalAgentNFSP(t_prof=self._t_prof)
        w = self.pull_current_eval_strategy()
        eval_agent.update_weights(weights_for_eval_agent=w)
        eval_agent.notify_of_reset()
        eval_agent.set_mode(EvalAgentNFSP.EVAL_MODE_AVG)
        eval_agent.store_to_disk(path=_dir, file_name="eval_agent")

    def checkpoint(self, **kwargs):
        pass

    def load_checkpoint(self, **kwargs):
        pass
