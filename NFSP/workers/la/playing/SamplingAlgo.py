import numpy as np
import torch


class SamplingAlgo:

    def __init__(self, t_prof, env_bldr, n_envs_avg, n_envs_br, br_buf2, avg_buf2, br_learner2, avg_learner2):
        if t_prof.nn_type == "recurrent":
            from PokerRL.rl.buffers.BRMemorySaverRNN import BRMemorySaverRNN
            from NFSP.workers.la.action_buffer.ActionBufferRNN import AvgMemorySaverRNN

            BR_MEM_SAVER = BRMemorySaverRNN
            AVG_MEM_SAVER = AvgMemorySaverRNN

        elif t_prof.nn_type == "feedforward":
            from PokerRL.rl.buffers.BRMemorySaverFLAT import BRMemorySaverFLAT
            from NFSP.workers.la.action_buffer.ActionBufferFLAT import AvgMemorySaverFLAT

            BR_MEM_SAVER = BRMemorySaverFLAT
            AVG_MEM_SAVER = AvgMemorySaverFLAT
        else:
            raise ValueError(t_prof.nn_type)
        self._t_prof = t_prof
        self._env_bldr = env_bldr
        self._antic = self._t_prof.antic_start
        self._br_buf2 = br_buf2
        self._avg_buf2 = avg_buf2
        self._br_learner2 = br_learner2
        self._avg_learner2 = avg_learner2

        self._avg_memory_savers = [
            [
                AVG_MEM_SAVER(env_bldr=self._env_bldr, buffer=self._avg_buf2[p])
                for _ in range(n_envs_avg)
            ]
            for p in range(self._env_bldr.N_SEATS)
        ]
        self._br_memory_savers = [
            [
                BR_MEM_SAVER(env_bldr=self._env_bldr, buffer=self._br_buf2[p])
                for _ in range(n_envs_br)
            ]
            for p in range(self._env_bldr.N_SEATS)
        ]

    @property
    def antic(self):
        return self._antic

    @antic.setter
    def antic(self, value):
        self._antic = value

    def play(self, nfsp_iter):
        raise NotImplementedError


class SeatActorBase:
    AVG = 1
    BR = 2

    @staticmethod
    def act_mixed(owner, current_policy_tags, step_wrappers, br_learner, avg_learner, random_prob):
        """ play with p*eps*rnd + p*(1-eps)*br and (1-p)*avg policy """

        with torch.no_grad():

            # """"""""""""""""""""""""
            # Construct
            # """"""""""""""""""""""""
            _sw_list_AVG = []
            _sw_list_BR = []

            for sw in step_wrappers:
                if current_policy_tags[sw.env_idx] == SeatActorBase.AVG:
                    _sw_list_AVG.append(sw)
                elif current_policy_tags[sw.env_idx] == SeatActorBase.BR:
                    _sw_list_BR.append(sw)
                else:
                    raise ValueError(current_policy_tags[sw.env_idx])

            # """"""""""""""""""""""""
            # AVG actions
            # """"""""""""""""""""""""
            SeatActorBase.act_avg(owner=owner, step_wrappers=_sw_list_AVG, avg_learner=avg_learner)

            # """"""""""""""""""""""""
            # BR actions
            # """"""""""""""""""""""""
            if random_prob > 0:
                SeatActorBase.act_eps_greedy(owner=owner, step_wrappers=_sw_list_BR, br_learner=br_learner,
                                             random_prob=random_prob)
            else:
                SeatActorBase.act_greedy(owner=owner, step_wrappers=_sw_list_BR, br_learner=br_learner)

    @staticmethod
    def act_constant_eps_greedy(owner, step_wrappers, br_learner):
        """ BR + eps """
        with torch.no_grad():
            if len(step_wrappers) > 0:
                actions, was_rnd = SeatActorBase.choose_a_br(br_learner=br_learner, owner=owner,
                                                             step_wrappers=step_wrappers, random_prob=br_learner.eps)
                for i, sw in enumerate(step_wrappers):
                    sw.action = actions[i].item()
                    sw.action_was_random = was_rnd

    @staticmethod
    def act_eps_greedy(owner, step_wrappers, br_learner, random_prob=None):
        """ BR + eps """
        with torch.no_grad():
            if len(step_wrappers) > 0:
                actions, was_rnd = SeatActorBase.choose_a_br(br_learner=br_learner, owner=owner,
                                                             step_wrappers=step_wrappers,
                                                             random_prob=br_learner.eps if random_prob is None else random_prob)
                for i, sw in enumerate(step_wrappers):
                    sw.action = actions[i].item()
                    sw.action_was_random = was_rnd

    @staticmethod
    def act_greedy(owner, step_wrappers, br_learner):
        """ BR + eps """
        with torch.no_grad():
            if len(step_wrappers) > 0:
                actions, was_rnd = SeatActorBase.choose_a_br(br_learner=br_learner, owner=owner,
                                                             step_wrappers=step_wrappers, random_prob=0)
                for i, sw in enumerate(step_wrappers):
                    sw.action = actions[i].item()
                    sw.action_was_random = was_rnd

    @staticmethod
    def act_avg(owner, step_wrappers, avg_learner):
        if len(step_wrappers) > 0:
            a_probs = avg_learner.get_a_probs(
                pub_obses=[sw.obs for sw in step_wrappers],
                range_idxs=np.array([sw.range_idxs[owner] for sw in step_wrappers], dtype=np.int32),
                legal_actions_lists=[sw.legal_actions_list for sw in step_wrappers],

            )
            _n_actions_arranged = np.arange(a_probs.shape[-1])
            for i, sw in enumerate(step_wrappers):
                sw.action = np.random.choice(
                    a=_n_actions_arranged,
                    p=a_probs[i],
                    replace=True
                ).item()
                sw.action_was_random = False

    @staticmethod
    def choose_a_br(owner, br_learner, step_wrappers, random_prob):
        """
        TODO maybe allow some explore some BR

        Returns:
            actions, was_random?:
        """
        pub_obses = [sw.obs for sw in step_wrappers]
        range_idxs = [sw.range_idxs[owner] for sw in step_wrappers]
        legal_actions_lists = [sw.legal_actions_list for sw in step_wrappers]

        # """""""""""""""""""""
        # Perhaps explore
        # """""""""""""""""""""
        if random_prob > np.random.random():
            actions = np.array([
                l[np.random.randint(low=0, high=len(l))]
                for l in legal_actions_lists
            ])
            return actions, True

        with torch.no_grad():
            # """""""""""""""""""""
            # Play by BR
            # """""""""""""""""""""
            actions = br_learner.select_br_a(
                pub_obses=pub_obses,
                range_idxs=range_idxs,
                legal_actions_lists=legal_actions_lists,
            )
            return actions, False

    @staticmethod
    def pick_training_policy(br_prob):
        if br_prob < np.random.random():
            return SeatActorBase.AVG
        return SeatActorBase.BR
