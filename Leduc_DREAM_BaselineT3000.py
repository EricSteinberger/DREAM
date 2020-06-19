import numpy as np

from DREAM_and_DeepCFR.TrainingProfile import TrainingProfile
from DREAM_and_DeepCFR.workers.driver.Driver import Driver
from HYPERS import *
from PokerRL.game.games import StandardLeduc  # or any other game

if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(
        name="SD-Leduc_DREAM_BaselineT3000_v001_SEED" + str(np.random.randint(1000000)),
        nn_type="feedforward",

        n_batches_adv_training=SDCFR_LEDUC_BATCHES,
        periodic_restart=SDCFR_LEDUC_PERIOD,
        n_traversals_per_iter=SDCFR_LEDUC_TRAVERSALS_OS,
        sampler="learned_baseline",
        n_batches_per_iter_baseline=3000,

        os_eps=OS_EPS,
        game_cls=StandardLeduc,

        DISTRIBUTED=False,
    ),
        eval_methods={
            "br": 3,
        })
    ctrl.run()
