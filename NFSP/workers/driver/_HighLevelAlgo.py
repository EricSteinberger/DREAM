# Copyright (c) 2019 Eric Steinberger


import time

from PokerRL.rl.base_cls.HighLevelAlgoBase import HighLevelAlgoBase as _HighLevelAlgoBase


class HighLevelAlgo(_HighLevelAlgoBase):

    def __init__(self, t_prof, la_handles, ps_handles, chief_handle):
        super().__init__(t_prof=t_prof, la_handles=la_handles, chief_handle=chief_handle)
        self.ps_handles = ps_handles
        self.ddqn_args = t_prof.module_args["ddqn"]
        self.n_br_updates = None

    def init(self):
        self._update_las(update_br=True, update_avg=True, update_eps=True, update_antic=True)
        self._update_all_target_nets()
        self.n_br_updates = 0

    def run_one_iter(self, n_br_updates, n_avg_updates, nfsp_iter):
        t_computation = 0.0
        t_syncing = 0.0
        t_playing = 0.0

        # _____________________________________________________ Play ___________________________________________________
        t_last = time.time()
        self.play(nfsp_iter=nfsp_iter)
        t_playing += time.time() - t_last

        # _________________________________________________ Do Q updates _______________________________________________
        for i in range(n_br_updates):
            #  parallel: compute and apply gradients
            t_last = time.time()
            self._apply_grads(br_or_avg="br", grads_all_p=self._get_grads("br"))
            t_computation += time.time() - t_last

            #  update q on all las
            t_last = time.time()
            self._update_las(update_br=True, update_avg=False, update_eps=False, update_antic=False)

            # counts over multiple global iterations for target_update_freqs > q_updates_per_iter
            self.n_br_updates += 1

            #  periodically update target network
            if self.n_br_updates % self.ddqn_args.target_net_update_freq == 0:
                self._update_all_target_nets()

            t_syncing += time.time() - t_last

        # ________________________________________________ Do Pi updates _______________________________________________
        for i in range(n_avg_updates):
            #  parallel: compute and apply gradients
            t_last = time.time()
            self._apply_grads(br_or_avg="avg", grads_all_p=self._get_grads("avg"))
            t_computation += time.time() - t_last

            # update pi on all las
            t_last = time.time()
            self._update_las(update_br=False, update_avg=True, update_eps=False, update_antic=False)
            t_syncing += time.time() - t_last

        # _________________________________________________ Increment  _________________________________________________
        _l = [self._ray.remote(self.ps_handles[p_id].increment)
              for p_id in range(self._t_prof.n_seats)]

        self._ray.wait(_l)
        self._update_las(update_br=True, update_avg=True, update_eps=True, update_antic=True)

        return {
            "t_playing": t_playing,
            "t_computation": t_computation,
            "t_syncing": t_syncing,
        }

    def play(self, nfsp_iter):
        self._ray.wait([
            self._ray.remote(la.play,
                             nfsp_iter)
            for la in self._la_handles
        ])

    def state_dict(self):
        return {
            "n_br_updates": self.n_br_updates
        }

    def load_state_dict(self, state):
        self.n_br_updates = state["n_br_updates"]

    def _update_las(self, update_br=True, update_avg=True, update_eps=True, update_antic=True):
        la_batches = []
        n = len(self._la_handles)
        c = 0
        while n > c:
            s = min(n, c + self._t_prof.max_n_las_sync_simultaneously)
            la_batches.append(self._la_handles[c:s])
            if type(la_batches[-1]) is not list:
                la_batches[-1] = [la_batches[-1]]
            c = s

        w_q = [None for _ in range(self._t_prof.n_seats)]
        w_avg = [None for _ in range(self._t_prof.n_seats)]
        eps = [None for _ in range(self._t_prof.n_seats)]
        antic = [None for _ in range(self._t_prof.n_seats)]

        for p_id in range(self._t_prof.n_seats):
            w_q[p_id] = None if not update_br else self._ray.remote(self.ps_handles[p_id].get_q1_weights)
            w_avg[p_id] = None if not update_avg else self._ray.remote(self.ps_handles[p_id].get_avg_weights)
            eps[p_id] = None if not update_eps else self._ray.remote(self.ps_handles[p_id].get_eps)
            antic[p_id] = None if not update_antic else self._ray.remote(self.ps_handles[p_id].get_antic)

        for batch in la_batches:
            _l = []
            for p_id in range(self._t_prof.n_seats):
                _l += [self._ray.remote(la.update,
                                        p_id, w_q[p_id], w_avg[p_id], eps[p_id], antic[p_id])
                       for la in batch]
            self._ray.wait(_l)

    def _update_all_target_nets(self):
        _l = []
        for p_id in range(self._t_prof.n_seats):
            _l += [
                self._ray.remote(la.update_q2,
                                 p_id)
                for la in self._la_handles
            ]
        self._ray.wait(_l)

    def _apply_grads(self, br_or_avg, grads_all_p):
        if br_or_avg == "br":
            self._ray.wait([
                self._ray.remote(self.ps_handles[p_id].apply_grads_br,
                                 grads_all_p[p_id])
                for p_id in range(self._t_prof.n_seats)
            ])

        elif br_or_avg == "avg":
            self._ray.wait([
                self._ray.remote(self.ps_handles[p_id].apply_grads_avg,
                                 grads_all_p[p_id])
                for p_id in range(self._t_prof.n_seats)
            ])

        else:
            raise ValueError(br_or_avg)

    def _empty_cir_bufs(self):
        self._ray.wait([
            self._ray.remote(la.empty_cir_bufs)
            for la in self._la_handles
        ])

    def _get_grads(self, br_or_avg):
        grads = {}
        for p_id in range(self._t_prof.n_seats):
            grads[p_id] = []
            for la in self._la_handles:
                if br_or_avg == "br":
                    g = self._ray.remote(la.get_br_grads,
                                         p_id)
                elif br_or_avg == "avg":
                    g = self._ray.remote(la.get_avg_grads,
                                         p_id)
                else:
                    raise ValueError(br_or_avg)

                grads[p_id].append(g)

        for p_id in range(self._t_prof.n_seats):
            self._ray.wait(grads[p_id])

        return grads
