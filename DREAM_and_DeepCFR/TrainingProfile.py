# Copyright (c) 2019 Eric Steinberger


import copy

import torch

from DREAM_and_DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DREAM_and_DeepCFR.workers.la.sampling_algorithms.MainPokerModuleFLAT_Baseline import MPMArgsFLAT_Baseline
from DREAM_and_DeepCFR.workers.la.AdvWrapper import AdvTrainingArgs
from DREAM_and_DeepCFR.workers.la.AvrgWrapper import AvrgTrainingArgs
from DREAM_and_DeepCFR.workers.la.sampling_algorithms.LearnedBaselineLearner import BaselineArgs
from PokerRL.game import bet_sets
from PokerRL.game.games import DiscretizedNLLeduc
from PokerRL.game.wrappers import FlatLimitPokerEnvBuilder
from PokerRL.rl.base_cls.TrainingProfileBase import TrainingProfileBase
from PokerRL.rl.neural.AvrgStrategyNet import AvrgNetArgs
from PokerRL.rl.neural.DuelingQNet import DuelingQArgs


class TrainingProfile(TrainingProfileBase):

    def __init__(self,

                 # ------ General
                 name="",
                 log_verbose=False,
                 log_memory=False,
                 log_export_freq=1,
                 checkpoint_freq=99999999,
                 eval_agent_export_freq=999999999,
                 n_learner_actor_workers=8,
                 max_n_las_sync_simultaneously=10,
                 nn_type="feedforward",  # "recurrent" or "feedforward"

                 # ------ Computing
                 path_data=None,
                 local_crayon_server_docker_address="localhost",
                 device_inference="cpu",
                 device_training="cpu",
                 device_parameter_server="cpu",
                 DISTRIBUTED=False,
                 CLUSTER=False,
                 DEBUGGING=False,

                 # ------ Env
                 game_cls=DiscretizedNLLeduc,
                 n_seats=2,
                 agent_bet_set=bet_sets.B_2,
                 start_chips=None,
                 chip_randomness=(0, 0),
                 uniform_action_interpolation=False,
                 use_simplified_headsup_obs=True,

                 # ------ Evaluation
                 eval_modes_of_algo=(EvalAgentDeepCFR.EVAL_MODE_SINGLE,),
                 eval_stack_sizes=None,

                 # ------ General Deep CFR params
                 n_traversals_per_iter=30000,
                 iter_weighting_exponent=1.0,
                 n_actions_traverser_samples=3,

                 sampler="mo",
                 turn_off_baseline=False,  # Only for VR-OS
                 os_eps=1,
                 periodic_restart=1,

                 # --- Baseline Hyperparameters
                 max_buffer_size_baseline=2e5,
                 batch_size_baseline=512,
                 n_batches_per_iter_baseline=300,

                 dim_baseline=64,
                 deep_baseline=True,
                 normalize_last_layer_FLAT_baseline=True,

                 # --- Adv Hyperparameters
                 n_batches_adv_training=5000,
                 init_adv_model="random",
                 mini_batch_size_adv=2048,
                 dim_adv=64,
                 deep_adv=True,
                 optimizer_adv="adam",
                 loss_adv="weighted_mse",
                 lr_adv=0.001,
                 grad_norm_clipping_adv=1.0,
                 lr_patience_adv=999999999,
                 normalize_last_layer_FLAT_adv=True,

                 max_buffer_size_adv=2e6,

                 # ------ SPECIFIC TO AVRG NET
                 n_batches_avrg_training=15000,
                 init_avrg_model="random",
                 dim_avrg=64,
                 deep_avrg=True,

                 mini_batch_size_avrg=2048,
                 loss_avrg="weighted_mse",
                 optimizer_avrg="adam",
                 lr_avrg=0.001,
                 grad_norm_clipping_avrg=1.0,
                 lr_patience_avrg=999999999,
                 normalize_last_layer_FLAT_avrg=True,

                 max_buffer_size_avrg=2e6,

                 # ------ SPECIFIC TO SINGLE
                 export_each_net=False,
                 eval_agent_max_strat_buf_size=None,

                 # ------ Optional
                 lbr_args=None,
                 rlbr_args=None,
                 h2h_args=None,

                 ):
        if nn_type == "feedforward":
            env_bldr_cls = FlatLimitPokerEnvBuilder

            from PokerRL.rl.neural.MainPokerModuleFLAT import MPMArgsFLAT

            mpm_args_adv = MPMArgsFLAT(deep=deep_adv, dim=dim_adv, normalize=normalize_last_layer_FLAT_adv)
            mpm_args_baseline = MPMArgsFLAT_Baseline(deep=deep_baseline, dim=dim_baseline,
                                                     normalize=normalize_last_layer_FLAT_baseline)
            mpm_args_avrg = MPMArgsFLAT(deep=deep_avrg, dim=dim_avrg, normalize=normalize_last_layer_FLAT_avrg)

        else:
            raise ValueError(nn_type)

        super().__init__(
            name=name,
            log_verbose=log_verbose,
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
                "adv_training": AdvTrainingArgs(
                    adv_net_args=DuelingQArgs(
                        mpm_args=mpm_args_adv,
                        n_units_final=dim_adv
                    ),
                    n_batches_adv_training=n_batches_adv_training,
                    init_adv_model=init_adv_model,
                    batch_size=mini_batch_size_adv,
                    optim_str=optimizer_adv,
                    loss_str=loss_adv,
                    lr=lr_adv,
                    grad_norm_clipping=grad_norm_clipping_adv,
                    device_training=device_training,
                    max_buffer_size=max_buffer_size_adv,
                    lr_patience=lr_patience_adv,
                ),
                "avrg_training": AvrgTrainingArgs(
                    avrg_net_args=AvrgNetArgs(
                        mpm_args=mpm_args_avrg,
                        n_units_final=dim_avrg,
                    ),
                    n_batches_avrg_training=n_batches_avrg_training,
                    init_avrg_model=init_avrg_model,
                    batch_size=mini_batch_size_avrg,
                    loss_str=loss_avrg,
                    optim_str=optimizer_avrg,
                    lr=lr_avrg,
                    grad_norm_clipping=grad_norm_clipping_avrg,
                    device_training=device_training,
                    max_buffer_size=max_buffer_size_avrg,
                    lr_patience=lr_patience_avrg,
                ),
                "env": game_cls.ARGS_CLS(
                    n_seats=n_seats,
                    starting_stack_sizes_list=[start_chips for _ in range(n_seats)],
                    bet_sizes_list_as_frac_of_pot=copy.deepcopy(agent_bet_set),
                    stack_randomization_range=chip_randomness,
                    use_simplified_headsup_obs=use_simplified_headsup_obs,
                    uniform_action_interpolation=uniform_action_interpolation
                ),
                "mccfr_baseline": BaselineArgs(
                    q_net_args=DuelingQArgs(
                        mpm_args=mpm_args_baseline,
                        n_units_final=dim_baseline,
                    ),
                    max_buffer_size=max_buffer_size_baseline,
                    batch_size=batch_size_baseline,
                    n_batches_per_iter_baseline=n_batches_per_iter_baseline,
                ),
                "lbr": lbr_args,
                "rlbr": rlbr_args,
                "h2h": h2h_args,
            },
            log_memory=log_memory,
        )

        self.nn_type = nn_type
        self.n_traversals_per_iter = int(n_traversals_per_iter)
        self.iter_weighting_exponent = iter_weighting_exponent
        self.sampler = sampler
        self.os_eps = os_eps
        self.periodic_restart = periodic_restart
        self.turn_off_baseline = turn_off_baseline
        self.n_actions_traverser_samples = n_actions_traverser_samples

        # SINGLE
        self.export_each_net = export_each_net
        self.eval_agent_max_strat_buf_size = eval_agent_max_strat_buf_size

        # Different for dist and local
        if DISTRIBUTED or CLUSTER:
            print("Running with ", n_learner_actor_workers, "LearnerActor Workers.")
            self.n_learner_actors = n_learner_actor_workers
        else:
            self.n_learner_actors = 1
        self.max_n_las_sync_simultaneously = max_n_las_sync_simultaneously

        assert isinstance(device_parameter_server, str), "Please pass a string (either 'cpu' or 'cuda')!"
        self.device_parameter_server = torch.device(device_parameter_server)
