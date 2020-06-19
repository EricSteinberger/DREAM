import numpy as np

from NFSP.TrainingProfile import TrainingProfile
from NFSP.workers.driver.Driver import Driver
from PokerRL import StandardLeduc

if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(
        name="Leduc_NFSP_v001_SEED" + str(np.random.randint(1000000)),
        game_cls=StandardLeduc,
        n_steps_per_iter_per_la=128,
        target_net_update_freq=300,
        min_prob_add_res_buf=0,
        lr_avg=0.01,
        lr_br=0.1,
    ),
        eval_methods={"br": 10000},
        n_iterations=None)
    ctrl.run()
