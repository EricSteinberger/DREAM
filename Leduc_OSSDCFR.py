import numpy as np

from DREAM_and_DeepCFR.TrainingProfile import TrainingProfile
from DREAM_and_DeepCFR.workers.driver.Driver import Driver
from HYPERS import *
from PokerRL.game.games import StandardLeduc  # or any other game

if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(
        name="Leduc_OSSDCFR_v001_SEED" + str(np.random.randint(1000000)),
        n_batches_adv_training=SDCFR_LEDUC_BATCHES,
        n_traversals_per_iter=SDCFR_LEDUC_TRAVERSALS_OS,

        sampler="mo",
        n_actions_traverser_samples=1,

        os_eps=OS_EPS,

        game_cls=StandardLeduc,

        DISTRIBUTED=False,
    ),
        eval_methods={
            "br": 20,
        })
    ctrl.run()
