# Copyright (c) 2019 Eric Steinberger


import copy

import torch

from NFSP.AvgWrapper import AvgWrapperArgs
from NFSP.EvalAgentNFSP import EvalAgentNFSP
from PokerRL.game import bet_sets
from PokerRL.game.games import StandardLeduc
from PokerRL.game.wrappers import FlatLimitPokerEnvBuilder
from PokerRL.rl.agent_modules import DDQNArgs
from PokerRL.rl.base_cls.TrainingProfileBase import TrainingProfileBase
from PokerRL.rl.neural.AvrgStrategyNet import AvrgNetArgs
from PokerRL.rl.neural.DuelingQNet import DuelingQArgs
from PokerRL.rl.neural.MainPokerModuleFLAT import MPMArgsFLAT


class TrainingProfile(TrainingProfileBase):

    def __init__(self,

                 # --- general
                 name,
                 log_export_freq=1000,
                 checkpoint_freq=99999999,
                 eval_agent_export_freq=99999999,

                 # --- Computing
                 path_data=None,
                 local_crayon_server_docker_address="localhost",
                 device_inference="cpu",
                 device_parameter_server="cpu",
                 n_learner_actor_workers=8,
                 max_n_las_sync_simultaneously=100,
                 DISTRIBUTED=False,
                 CLUSTER=False,
                 DEBUGGING=False,
                 VERBOSE=True,

                 # --- env
                 game_cls=StandardLeduc,
                 n_seats=2,
                 use_simplified_headsup_obs=True,
                 start_chips=None,

                 agent_bet_set=bet_sets.B_2,
                 stack_randomization_range=(0, 0),
                 uniform_action_interpolation=False,

                 # --- Evaluation
                 eval_modes_of_algo=(EvalAgentNFSP.EVAL_MODE_AVG,),
                 eval_stack_sizes=None,

                 # --- NFSP
                 nn_type="feedforward",
                 linear=False,
                 sampling=None,  # "adam" or "clean" for enhanced variants

                 # Original NFSP also adds epsilon-exploration actions to the averaging buffer.
                 add_random_actions_to_avg_buffer=True,

                 n_br_updates_per_iter=2,
                 n_avg_updates_per_iter=2,
                 target_net_update_freq=300,  # every N neural net updates. Not every N global iters, episodes, or steps
                 cir_buf_size_each_la=2e5,
                 res_buf_size_each_la=2e6,  # the more the better to infinity
                 min_prob_add_res_buf=0.0,  # 0.0 =  vanilla reservoir; >0 exponential averaging.

                 eps_start=0.06,
                 eps_const=0.01,
                 eps_exponent=0.5,
                 eps_min=0.0,

                 antic_start=0.1,
                 antic_const=0.01,
                 antic_exponent=0.5,
                 antic_min=0.1,  # No decay

                 # For clean sampling
                 constant_eps_expl=0.1,

                 # --- Training.
                 n_steps_avg_per_iter_per_la=None,
                 n_steps_br_per_iter_per_la=None,
                 n_steps_per_iter_per_la=None,

                 training_multiplier_iter_0=1,  # In iter 0 the BR net is clueless, but adds to res_buf. -> "pretrain"

                 # --- Q-Learning Hyperparameters
                 mini_batch_size_br_per_la=128,
                 dim_br=64,
                 normalize_last_layer_flat=True,
                 lr_br=0.1,
                 deep_br=True,  # True -> Use deep multi-branch network; False -> Use shallow net
                 grad_norm_clipping_br=1.0,
                 optimizer_br="sgd",
                 loss_br="mse",

                 # --- Avg Network Hyperparameters
                 mini_batch_size_avg_per_la=128,
                 dim_avg=64,
                 lr_avg=0.005,
                 deep_avg=True,  # True -> Use deep multi-branch network; False -> Use shallow net
                 grad_norm_clipping_avg=1.0,
                 optimizer_avg="sgd",
                 loss_avg="weighted_ce",

                 # Option
                 lbr_args=None,
                 rlbr_args=None,
                 ):
        if nn_type == "recurrent":
            raise NotImplementedError

        elif nn_type == "feedforward":
            env_bldr_cls = FlatLimitPokerEnvBuilder

            mpm_args_br = MPMArgsFLAT(dim=dim_br, normalize=normalize_last_layer_flat, deep=deep_br)
            mpm_args_avg = MPMArgsFLAT(dim=dim_avg, normalize=normalize_last_layer_flat, deep=deep_avg)

        else:
            raise ValueError(nn_type)

        super().__init__(

            name=name,
            log_verbose=VERBOSE,
            log_export_freq=log_export_freq,
            checkpoint_freq=checkpoint_freq,
            eval_agent_export_freq=eval_agent_export_freq,
            path_data=path_data,
            game_cls=game_cls,
            env_bldr_cls=env_bldr_cls,
            start_chips=start_chips,
            eval_modes_of_algo=eval_modes_of_algo,
            eval_stack_sizes=eval_stack_sizes,

            DEBUGGING=DEBUGGING,
            DISTRIBUTED=DISTRIBUTED,
            CLUSTER=CLUSTER,
            device_inference=device_inference,
            local_crayon_server_docker_address=local_crayon_server_docker_address,

            module_args={
                "ddqn": DDQNArgs(
                    q_args=DuelingQArgs(
                        mpm_args=mpm_args_br,
                        n_units_final=dim_br,
                    ),
                    cir_buf_size=int(cir_buf_size_each_la),
                    batch_size=mini_batch_size_br_per_la,
                    target_net_update_freq=target_net_update_freq,
                    optim_str=optimizer_br,
                    loss_str=loss_br,
                    lr=lr_br,
                    eps_start=eps_start,
                    eps_const=eps_const,
                    eps_exponent=eps_exponent,
                    eps_min=eps_min,
                    grad_norm_clipping=grad_norm_clipping_br,
                ),
                "avg": AvgWrapperArgs(
                    avg_net_args=AvrgNetArgs(
                        mpm_args=mpm_args_avg,
                        n_units_final=dim_avg,
                    ),
                    batch_size=mini_batch_size_avg_per_la,
                    res_buf_size=int(res_buf_size_each_la),
                    min_prob_add_res_buf=min_prob_add_res_buf,
                    loss_str=loss_avg,
                    optim_str=optimizer_avg,
                    lr=lr_avg,
                    grad_norm_clipping=grad_norm_clipping_avg,
                ),
                "env": game_cls.ARGS_CLS(
                    n_seats=n_seats,
                    starting_stack_sizes_list=[start_chips for _ in range(n_seats)],
                    stack_randomization_range=stack_randomization_range,
                    use_simplified_headsup_obs=use_simplified_headsup_obs,
                    uniform_action_interpolation=uniform_action_interpolation,

                    # Set up in a way that just ignores this if not DiscretizedLimit
                    bet_sizes_list_as_n_chips=copy.deepcopy(agent_bet_set),

                    # Set up in a way that just ignores this if not Discretized
                    bet_sizes_list_as_frac_of_pot=copy.deepcopy(agent_bet_set),
                ),
                "lbr": lbr_args,
                "rlbr": rlbr_args,
            },
            log_memory=False,
        )

        # ____________________________________________________ NFSP ____________________________________________________
        self.nn_type = nn_type
        self.n_br_updates_per_iter = int(n_br_updates_per_iter)
        self.n_avg_updates_per_iter = int(n_avg_updates_per_iter)
        self.antic_start = antic_start
        self.antic_const = antic_const
        self.antic_exponent = antic_exponent
        self.antic_min = antic_min
        self.linear = linear
        self.sampling = sampling
        self.add_random_actions_to_buffer = add_random_actions_to_avg_buffer
        self.training_multiplier_iter_0 = int(training_multiplier_iter_0)
        self.constant_eps_expl = constant_eps_expl

        if sampling == "adam" or sampling == "clean":
            assert n_steps_per_iter_per_la is None
            assert n_steps_br_per_iter_per_la is not None
            assert n_steps_avg_per_iter_per_la is not None
            self.n_steps_br_per_iter_per_la = int(n_steps_br_per_iter_per_la)
            self.n_steps_avg_per_iter_per_la = int(n_steps_avg_per_iter_per_la)
        else:
            assert n_steps_per_iter_per_la is not None
            assert n_steps_br_per_iter_per_la is None
            assert n_steps_avg_per_iter_per_la is None
            self.n_steps_per_iter_per_la = int(n_steps_per_iter_per_la)

        if DISTRIBUTED or CLUSTER:
            self.n_learner_actors = int(n_learner_actor_workers)
        else:
            self.n_learner_actors = 1

        self.max_n_las_sync_simultaneously = int(max_n_las_sync_simultaneously)

        assert isinstance(device_parameter_server, str), "Please pass a string (either 'cpu' or 'cuda')!"
        self.device_parameter_server = torch.device(device_parameter_server)
