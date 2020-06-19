import numpy as np

from DREAM_and_DeepCFR.EvalAgentDeepCFR import EvalAgentDeepCFR
from DREAM_and_DeepCFR.TrainingProfile import TrainingProfile
from DREAM_and_DeepCFR.workers.driver.Driver import Driver
from HYPERS import *
from PokerRL.game.games import StandardLeduc  # or any other game

if __name__ == '__main__':
    ctrl = Driver(t_prof=TrainingProfile(
        name="Leduc_DREAM_v001_SEED" + str(np.random.randint(1000000)),
        nn_type="feedforward",

        n_batches_adv_training=SDCFR_LEDUC_BATCHES,
        n_traversals_per_iter=SDCFR_LEDUC_TRAVERSALS_OS,
        sampler="learned_baseline",
        n_batches_per_iter_baseline=SDCFR_LEDUC_BASELINE_BATCHES,

        os_eps=OS_EPS,
        game_cls=StandardLeduc,

        eval_modes_of_algo=(
            EvalAgentDeepCFR.EVAL_MODE_SINGLE,
            EvalAgentDeepCFR.EVAL_MODE_AVRG_NET
        ),

        n_batches_avrg_training=4000,

        DISTRIBUTED=False,
    ),
        eval_methods={
            "br": 20,
        })
    ctrl.run()
