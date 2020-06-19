import numpy as np

from PokerRL.eval.rl_br import _util
from PokerRL.rl import rl_util
from PokerRL.rl.agent_modules.DDQN import DDQN
from PokerRL.rl.base_cls.workers.WorkerBase import WorkerBase


class Local_RLBR_LearnerActor(WorkerBase):

    def __init__(self, t_prof, chief_handle, eval_agent_cls):
        super().__init__(t_prof=t_prof)
        self._args = t_prof.module_args["rlbr"]

        self._env_bldr = rl_util.get_env_builder(t_prof=t_prof)

        self._chief_handle = chief_handle
        self._eval_agent_cls = eval_agent_cls
        self._eval_env_bldr = _util.get_env_builder_rlbr(t_prof=t_prof)

        self._ddqns = [None for _ in range(self._eval_env_bldr.N_SEATS)]
        self._rlbr_seat_id = None
        self._agent_seat_id = None
        self._rlbr_env_wrapper = None
        self._opponent = None
        self._buf = None
        self._br_memory_saver = None

        if t_prof.nn_type == "recurrent":
            from PokerRL.rl.buffers.CircularBufferRNN import CircularBufferRNN
            from PokerRL.rl.buffers.BRMemorySaverRNN import BRMemorySaverRNN

            self.CircularBufferCls = CircularBufferRNN
            self.BRMemorySaverCls = BRMemorySaverRNN
        elif t_prof.nn_type == "feedforward":
            from PokerRL.rl.buffers.CircularBufferFLAT import CircularBufferFLAT
            from PokerRL.rl.buffers.BRMemorySaverFLAT import BRMemorySaverFLAT

            self.CircularBufferCls = CircularBufferFLAT
            self.BRMemorySaverCls = BRMemorySaverFLAT

        else:
            raise ValueError(t_prof.nn_type)

    def reset(self, p_training, eval_opponent_state_dict, stack_size):
        self._rlbr_seat_id = p_training
        self._agent_seat_id = 1 - p_training
        self._opponent = self._eval_agent_cls(t_prof=self._t_prof)
        self._opponent.load_state_dict(eval_opponent_state_dict)
        self._rlbr_env_wrapper = self._eval_env_bldr.get_new_wrapper(is_evaluating=True, stack_size=stack_size)
        self._ddqns[p_training] = DDQN(owner=p_training, ddqn_args=self._args.ddqn_args,
                                       env_bldr=self._eval_env_bldr)
        self._buf = self.CircularBufferCls(env_bldr=self._env_bldr, max_size=self._args.ddqn_args.cir_buf_size)
        self._br_memory_saver = self.BRMemorySaverCls(env_bldr=self._eval_env_bldr, buffer=self._buf)

    def get_grads(self, p_id):
        return self._ray.grads_to_numpy(self._ddqns[p_id].get_grads_one_batch_from_buffer(buffer=self._buf))

    def play(self, n_episodes):
        self._ddqns[self._rlbr_seat_id].eval()
        accumulated_rew = 0.0
        for n in range(n_episodes):

            # """""""""""""""""
            # Reset
            # """""""""""""""""
            obs, r_for_all, done, info = self._rlbr_env_wrapper.reset()
            self._opponent.reset(deck_state_dict=self._rlbr_env_wrapper.env.cards_state_dict())

            range_idxs = [
                self._rlbr_env_wrapper.env.get_range_idx(p_id=p_id)
                for p_id in range(self._eval_env_bldr.N_SEATS)
            ]

            # Store last game to buffer and reset memory saver
            self._br_memory_saver.reset(range_idx=range_idxs[self._rlbr_seat_id])

            # """""""""""""""""
            # Play Episode
            # """""""""""""""""
            while not done:
                p_id_acting = self._rlbr_env_wrapper.env.current_player.seat_id
                if self._t_prof.DEBUGGING:
                    if p_id_acting != self._opponent._internal_env_wrapper.env.current_player.seat_id:
                        raise RuntimeError("Bad bad bug in RL-BR.")

                # RL-BR acting
                if p_id_acting == self._rlbr_seat_id:
                    legal_actions_list_t = self._rlbr_env_wrapper.env.get_legal_actions()
                    action_int = self._ddqns[self._rlbr_seat_id].select_br_a(
                        pub_obses=[obs],
                        range_idxs=np.array([range_idxs[self._rlbr_seat_id]], dtype=np.int32),
                        explore=True,
                        legal_actions_lists=[legal_actions_list_t]
                    )[0].item()

                    self._br_memory_saver.add_experience(obs_t_before_acted=obs,
                                                         a_selected_t=action_int,
                                                         legal_actions_list_t=legal_actions_list_t)

                    # Notify agent
                    self._opponent.notify_of_action(p_id_acted=self._rlbr_seat_id, action_he_did=action_int)

                # EvalAgent (opponent) acting
                else:
                    action_int, _ = self._opponent.get_action(step_env=True, need_probs=False)

                # Step
                obs, r_for_all, done, info = self._rlbr_env_wrapper.step(action=action_int)

            # Add terminal memory
            self._br_memory_saver.add_to_buf(reward_p=r_for_all[self._rlbr_seat_id], terminal_obs=obs)

            # For tracking running reward while training
            accumulated_rew += r_for_all[self._rlbr_seat_id]

        return accumulated_rew \
               * self._eval_env_bldr.env_cls.EV_NORMALIZER \
               * self._rlbr_env_wrapper.env.REWARD_SCALAR \
               / n_episodes

    def update_target_net(self, p_id):
        self._ddqns[p_id].update_target_net()

    def update(self, eps, nets):
        for p_id in range(self._t_prof.n_seats):
            if nets[p_id] is not None:
                self._ddqns[p_id].load_net_state_dict(
                    state_dict=self._ray.state_dict_to_torch(self._ray.get(nets[p_id]),
                                                             device=self._ddqns[p_id].device))

            if eps[p_id] is not None:
                self._ddqns[p_id].eps = self._ray.get(eps[p_id])

    # TODO this is duplicate code from RLBR LA
    def compute_rlbr(self, n_hands_each_seat, ddqn_state_dicts, stack_size):
        agent_losses = np.empty(shape=n_hands_each_seat * 2, dtype=np.float32)
        rlbr_dqn_each_seat = [
            DDQN.inference_version_from_state_dict(state_dict=ddqn_state_dicts[p],
                                                   env_bldr=self._eval_env_bldr)
            for p in range(self._t_prof.n_seats)
        ]
        rlbr_env_wrapper = self._eval_env_bldr.get_new_wrapper(is_evaluating=True, stack_size=stack_size)

        for rlbr_seat_id in range(rlbr_env_wrapper.env.N_SEATS):
            rlbr_agent = rlbr_dqn_each_seat[rlbr_seat_id]
            for iteration_id in range(n_hands_each_seat):

                # """""""""""""""""
                # Reset
                # """""""""""""""""
                obs, r_for_all, done, info = _util.reset_episode_multi_action_space(rlbr_env_wrapper=rlbr_env_wrapper,
                                                                                    opponent_agent=self._opponent)
                range_idx_rlbr = rlbr_env_wrapper.env.get_range_idx(p_id=rlbr_seat_id)

                # """""""""""""""""
                # Play Episode
                # """""""""""""""""
                while not done:
                    p_id_acting = rlbr_env_wrapper.env.current_player.seat_id

                    # RL-BR acting
                    if p_id_acting == rlbr_seat_id:
                        action_int = rlbr_agent.select_br_a(
                            pub_obses=[obs],
                            range_idxs=np.array([range_idx_rlbr], dtype=np.int32),
                            legal_actions_lists=[rlbr_env_wrapper.env.get_legal_actions()],
                            explore=False,
                        )[0]
                        self._opponent.notify_of_action(p_id_acted=rlbr_seat_id, action_he_did=action_int)

                    # EvalAgent (opponent) acting
                    else:
                        action_int, _ = self._opponent.get_action(step_env=True, need_probs=False)

                    # Step
                    obs, r_for_all, done, info = rlbr_env_wrapper.step(action=action_int)

                # add rews
                agent_losses[iteration_id + (rlbr_seat_id * n_hands_each_seat)] = r_for_all[rlbr_seat_id] \
                                                                                  * rlbr_env_wrapper.env.REWARD_SCALAR \
                                                                                  * rlbr_env_wrapper.env.EV_NORMALIZER

        return agent_losses.tolist()
