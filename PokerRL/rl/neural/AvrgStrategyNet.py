# Copyright (c) 2019 Eric Steinberger


import torch
import torch.nn as nn


class AvrgStrategyNet(nn.Module):

    def __init__(self, avrg_net_args, env_bldr, device):
        super().__init__()
        self.args = avrg_net_args
        self.env_bldr = env_bldr
        self.n_actions = self.env_bldr.N_ACTIONS

        MPM = avrg_net_args.mpm_args.get_mpm_cls()

        self._mpm = MPM(env_bldr=env_bldr, device=device, mpm_args=self.args.mpm_args)

        self._head = nn.Linear(in_features=self._mpm.output_units, out_features=self.n_actions)

        self.to(device)

    def forward(self, pub_obses, range_idxs, legal_action_masks):
        """
        Softmax is not applied in here! It is separate in training and action fns
        """

        y = self._mpm(pub_obses=pub_obses, range_idxs=range_idxs)
        y = self._head(y)
        y = torch.where(legal_action_masks == 1,
                        y,
                        torch.FloatTensor([-10e20]).to(device=y.device).expand_as(y))
        return y


class AvrgNetArgs:

    def __init__(self,
                 mpm_args,
                 n_units_final
                 ):
        self.mpm_args = mpm_args
        self.n_units_final = n_units_final
