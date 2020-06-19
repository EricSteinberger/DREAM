# Copyright (c) 2019 Eric Steinberger


import ray
import torch

from NFSP.workers.la.local import LearnerActor as _LocalLearnerActor


@ray.remote(num_cpus=1, num_gpus=1 if torch.cuda.is_available() else 0)
class LearnerActor(_LocalLearnerActor):

    def __init__(self, t_prof, worker_id, chief_handle):
        super().__init__(t_prof=t_prof, worker_id=worker_id, chief_handle=chief_handle)
