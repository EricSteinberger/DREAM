# Copyright (c) 2019 Eric Steinberger


import ray
import torch

from NFSP.workers.ps.local import ParameterServer as _LocalParameterServer


@ray.remote(num_cpus=1, num_gpus=1 if torch.cuda.is_available() else 0)
class ParameterServer(_LocalParameterServer):

    def __init__(self, t_prof, seat_id, chief_handle):
        super().__init__(t_prof=t_prof, seat_id=seat_id, chief_handle=chief_handle)
