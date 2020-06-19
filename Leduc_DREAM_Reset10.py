import numpy as np

from DREAM_and_DeepCFR.TrainingProfile import TrainingProfile
from DREAM_and_DeepCFR.workers.driver.Driver import Driver
from PokerRL.game.games import StandardLeduc  # or any other game

if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(
        name="Leduc_DREAM_Reset10_v001_SEED" + str(np.random.randint(1000000)),

        n_traversals_per_iter=900,

        n_batches_adv_training=3000,
        periodic_restart=10,
        sampler="learned_baseline",
        os_eps=0.5,
        game_cls=StandardLeduc,
    ),
        eval_methods={
            "br": 3,
        })
    ctrl.run()
