# Copyright (c) 2019 Eric Steinberger


import copy

from PokerRL.game import bet_sets
from PokerRL.game.poker_env_args import DiscretizedPokerEnvArgs, LimitPokerEnvArgs, NoLimitPokerEnvArgs
from PokerRL.rl.agent_modules.DDQN import DDQNArgs
from PokerRL.rl.neural.DuelingQNet import DuelingQArgs


class RLBRArgs:

    def __init__(self,
                 n_brs_to_train=1,
                 rlbr_bet_set=bet_sets.B_2,
                 n_hands_each_seat_per_la=20000,
                 n_las_per_player=3,

                 # Training
                 DISTRIBUTED=False,
                 n_iterations=10000,
                 play_n_steps_per_iter_per_la=256,
                 pretrain_n_steps=128,
                 device_training="cpu",

                 # the DDQN
                 nn_type="feedforward",
                 target_net_update_freq=200,
                 batch_size=512,
                 buffer_size=5e4,
                 optim_str="adam",
                 loss_str="mse",
                 lr=0.001,
                 eps_start=0.3,
                 eps_min=0.02,
                 eps_const=0.02,
                 eps_exponent=0.7,

                 # the QNet
                 dim=64,
                 deep=True,
                 normalize_last_layer_flat=True,
                 dropout=0.0,
                 ):
        if nn_type == "recurrent":
            raise NotImplementedError

        elif nn_type == "feedforward":
            from PokerRL.rl.neural.MainPokerModuleFLAT import MPMArgsFLAT

            mpm_args = MPMArgsFLAT(
                deep=deep,
                dim=dim,
                dropout=dropout,
                normalize=normalize_last_layer_flat,
            )

        else:
            raise ValueError(nn_type)

        if DISTRIBUTED and n_las_per_player < 1:
            raise RuntimeError("RL-BR needs at least 2 workers, when running distributed. This is for 1 ParameterServer"
                               "and at least one LearnerActor")

        self.n_brs_to_train = n_brs_to_train

        self.n_las_per_player = n_las_per_player if DISTRIBUTED else 1

        self.n_hands_each_seat_per_la = int(n_hands_each_seat_per_la)
        self.n_iterations = int(n_iterations)
        self.play_n_steps_per_iter_per_la = int(play_n_steps_per_iter_per_la)
        self.pretrain_n_steps = int(pretrain_n_steps)
        self.rlbr_bet_set = rlbr_bet_set

        self.ddqn_args = DDQNArgs(
            q_args=DuelingQArgs(
                n_units_final=dim,
                mpm_args=mpm_args),
            cir_buf_size=int(buffer_size),
            batch_size=int(batch_size),
            target_net_update_freq=target_net_update_freq,
            optim_str=optim_str,
            loss_str=loss_str,
            lr=lr,
            eps_start=eps_start,
            eps_const=eps_const,
            eps_exponent=eps_exponent,
            eps_min=eps_min,
            grad_norm_clipping=1.0,
            device_training=device_training,
        )

    def get_rlbr_env_args(self, agents_env_args, randomization_range=None):
        arg_cls = type(agents_env_args)

        if arg_cls is DiscretizedPokerEnvArgs:
            return DiscretizedPokerEnvArgs(
                n_seats=agents_env_args.n_seats,
                starting_stack_sizes_list=copy.deepcopy(agents_env_args.starting_stack_sizes_list),
                bet_sizes_list_as_frac_of_pot=copy.deepcopy(self.rlbr_bet_set),
                stack_randomization_range=randomization_range if randomization_range else (0, 0),
                use_simplified_headsup_obs=agents_env_args.use_simplified_headsup_obs,
                uniform_action_interpolation=False
            )

        elif arg_cls is LimitPokerEnvArgs:
            return LimitPokerEnvArgs(
                n_seats=agents_env_args.n_seats,
                starting_stack_sizes_list=copy.deepcopy(agents_env_args.starting_stack_sizes_list),
                stack_randomization_range=randomization_range if randomization_range else (0, 0),
                use_simplified_headsup_obs=agents_env_args.use_simplified_headsup_obs,
                uniform_action_interpolation=False
            )

        elif arg_cls is NoLimitPokerEnvArgs:
            raise NotImplementedError("Currently not supported")

        else:
            raise NotImplementedError(arg_cls)
