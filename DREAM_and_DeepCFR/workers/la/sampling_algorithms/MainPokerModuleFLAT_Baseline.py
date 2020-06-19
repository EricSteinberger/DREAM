# Copyright (c) 2019 Eric Steinberger


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PokerRL.rl.neural.CardEmbedding import CardEmbedding
from PokerRL.rl.neural.LayerNorm import LayerNorm


class MainPokerModuleFLAT_Baseline(nn.Module):

    def __init__(self,
                 env_bldr,
                 device,
                 mpm_args,
                 ):
        super().__init__()

        self._args = mpm_args
        self._env_bldr = env_bldr

        self._device = device

        self._board_start = self._env_bldr.obs_board_idxs[0]
        self._board_stop = self._board_start + len(self._env_bldr.obs_board_idxs)

        self.dropout = nn.Dropout(p=mpm_args.dropout)

        self.card_emb = CardEmbedding(env_bldr=env_bldr, dim=mpm_args.dim, device=device)

        if mpm_args.deep:
            self.cards_fc_1 = nn.Linear(in_features=self.card_emb.out_size * 2,
                                        out_features=mpm_args.dim * 3)
            self.cards_fc_2 = nn.Linear(in_features=mpm_args.dim * 3, out_features=mpm_args.dim * 3)
            self.cards_fc_3 = nn.Linear(in_features=mpm_args.dim * 3, out_features=mpm_args.dim)

            self.history_1 = nn.Linear(in_features=self._env_bldr.pub_obs_size - self._env_bldr.obs_size_board,
                                       out_features=mpm_args.dim)
            self.history_2 = nn.Linear(in_features=mpm_args.dim, out_features=mpm_args.dim)

            self.comb_1 = nn.Linear(in_features=2 * mpm_args.dim, out_features=mpm_args.dim)
            self.comb_2 = nn.Linear(in_features=mpm_args.dim, out_features=mpm_args.dim)

        else:
            self.layer_1 = nn.Linear(in_features=self.card_emb.out_size * 2
                                                 + self._env_bldr.pub_obs_size - self._env_bldr.obs_size_board,
                                     out_features=mpm_args.dim)
            self.layer_2 = nn.Linear(in_features=mpm_args.dim, out_features=mpm_args.dim)
            self.layer_3 = nn.Linear(in_features=mpm_args.dim, out_features=mpm_args.dim)

        if self._args.normalize:
            self.norm = LayerNorm(mpm_args.dim)

        self.to(device)
        # print("n parameters:", sum(p.numel() for p in self.parameters() if p.requires_grad))

    @property
    def output_units(self):
        return self._args.dim

    @property
    def device(self):
        return self._device

    def forward(self, pub_obses, range_idxs):
        """
        1. do list -> padded
        2. feed through pre-processing fc layers
        3. PackedSequence (sort, pack)
        4. rnn
        5. unpack (unpack re-sort)
        6. cut output to only last entry in sequence

        Args:
            pub_obses (list):                 list of np arrays of shape [np.arr([history_len, n_features]), ...)
            range_idxs (LongTensor):        range_idxs (one for each pub_obs) tensor([2, 421, 58, 912, ...])
        """
        if isinstance(pub_obses, list):
            pub_obses = torch.from_numpy(np.array(pub_obses)).to(self._device, torch.float32)

        hist_o = torch.cat([
            pub_obses[:, :self._board_start],
            pub_obses[:, self._board_stop:]
        ], dim=-1)

        # """""""""""""""""""""""
        # Card embeddings
        # """""""""""""""""""""""
        range_idxs_0 = range_idxs // 10000  # Big hack! See LearnedBaselineSampler for the reverse opp
        range_idxs_1 = range_idxs % 10000

        card_o_0 = self.card_emb(pub_obses=pub_obses,
                                 range_idxs=torch.where(range_idxs_0 == 8888, torch.zeros_like(range_idxs_0),
                                                        range_idxs_0))

        card_o_0 = torch.where(range_idxs_0.unsqueeze(1).expand_as(card_o_0) == 8888,
                               torch.full_like(card_o_0, fill_value=-1),
                               card_o_0,
                               )

        card_o_1 = self.card_emb(pub_obses=pub_obses,
                                 range_idxs=torch.where(range_idxs_1 == 8888, torch.zeros_like(range_idxs_1),
                                                        range_idxs_1))
        card_o_1 = torch.where(range_idxs_1.unsqueeze(1).expand_as(card_o_0) == 8888,
                               torch.full_like(card_o_1, fill_value=-1),
                               card_o_1,
                               )
        card_o = torch.cat([card_o_0, card_o_1], dim=-1)

        # """""""""""""""""""""""
        # Network
        # """""""""""""""""""""""
        if self._args.dropout > 0:
            A = lambda x: self.dropout(F.relu(x))
        else:
            A = lambda x: F.relu(x)

        if self._args.deep:
            card_o = A(self.cards_fc_1(card_o))
            card_o = A(self.cards_fc_2(card_o) + card_o)
            card_o = A(self.cards_fc_3(card_o))

            hist_o = A(self.history_1(hist_o))
            hist_o = A(self.history_2(hist_o) + hist_o)

            y = A(self.comb_1(torch.cat([card_o, hist_o], dim=-1)))
            y = A(self.comb_2(y) + y)

        else:
            y = torch.cat([hist_o, card_o], dim=-1)
            y = A(self.layer_1(y))
            y = A(self.layer_2(y) + y)
            y = A(self.layer_3(y) + y)

        # """""""""""""""""""""""
        # Normalize last layer
        # """""""""""""""""""""""
        if self._args.normalize:
            y = self.norm(y)

        return y


class MPMArgsFLAT_Baseline:

    def __init__(self,
                 deep=True,
                 dim=128,
                 dropout=0.0,
                 normalize=True,
                 ):
        self.deep = deep
        self.dim = dim
        self.dropout = dropout
        self.normalize = normalize

    def get_mpm_cls(self):
        return MainPokerModuleFLAT_Baseline
