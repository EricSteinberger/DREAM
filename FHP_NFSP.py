import numpy as np

from HYPERS import *
from NFSP.TrainingProfile import TrainingProfile
from NFSP.workers.driver.Driver import Driver
from PokerRL import Flop3Holdem

if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(
        name="FHP_NFSP_v001_SEED" + str(np.random.randint(1000000)),
        game_cls=Flop3Holdem,
        eps_const=0.005,
        eps_start=0.08,
        target_net_update_freq=1000,
        min_prob_add_res_buf=0.25,
        lr_avg=0.01,
        lr_br=0.1,

        n_learner_actor_workers=N_LA_FHP_NFSP,
        res_buf_size_each_la=int(2e7 / N_LA_FHP_NFSP),
        cir_buf_size_each_la=int(6e5 / N_LA_FHP_NFSP),
        n_steps_per_iter_per_la=int(256 / N_LA_FHP_NFSP),
        mini_batch_size_br_per_la=int(256 / N_LA_FHP_NFSP),
        mini_batch_size_avg_per_la=int(256 / N_LA_FHP_NFSP),

        DISTRIBUTED=True,
        rlbr_args=DIST_RLBR_ARGS_games,

    ),
        eval_methods={"rlbr": 200000},
        n_iterations=None)
    ctrl.run()
