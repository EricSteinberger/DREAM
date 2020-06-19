import numpy as np
import torch
from torch import nn


class CardEmbedding(nn.Module):

    def __init__(self, env_bldr, dim, device):
        super().__init__()
        self._env_bldr = env_bldr
        self._device = device

        self._board_start = env_bldr.obs_board_idxs[0]
        self._board_stop = self._board_start + len(env_bldr.obs_board_idxs)

        # preflop (counts hole cards), ...
        n_card_types = len(env_bldr.rules.ALL_ROUNDS_LIST)
        self.card_embs = nn.ModuleList([
            _CardGroupEmb(env_bldr=env_bldr, dim=dim)
            for _ in range(n_card_types)
        ])
        self._n_card_types = n_card_types
        self._dim = dim

        self._lut_range_idx_to_1d = env_bldr.lut_holder.LUT_IDX_2_HOLE_CARDS.astype(np.int64)
        self._lut_1d_to_2d = env_bldr.lut_holder.LUT_1DCARD_2_2DCARD.astype(np.int64)

        self.to(device)

    @property
    def out_size(self):
        return self._n_card_types * self._dim

    @property
    def device(self):
        return self._device

    def forward(self, pub_obses, range_idxs):
        priv_cards_1d = self._lut_range_idx_to_1d[range_idxs]

        priv_cards_2d = torch.from_numpy(self._lut_1d_to_2d[priv_cards_1d]) \
            .view(-1, self._env_bldr.rules.N_HOLE_CARDS, 2) \
            .to(device=self._device)

        priv_cards_1d = torch.from_numpy(priv_cards_1d) \
            .view(-1, self._env_bldr.rules.N_HOLE_CARDS) \
            .to(device=self._device)

        priv_ranks = priv_cards_2d[:, :, 0]
        priv_suits = priv_cards_2d[:, :, 1]

        board = pub_obses[:, self._board_start:self._board_stop].round().to(torch.long)
        card_batches = [(priv_ranks, priv_suits, priv_cards_1d)]
        off = 0
        for round_ in self._env_bldr.rules.ALL_ROUNDS_LIST:
            n = self._env_bldr.lut_holder.DICT_LUT_CARDS_DEALT_IN_TRANSITION_TO[round_]
            if n > 0:
                card_batches.append(
                    # rank, suit, card
                    (board[:, off:off + 3 * n:3],
                     board[:, off + 1:off + 1 + 3 * n:3],
                     board[:, off + 2:off + 2 + 3 * n:3],)
                )
                off += n

        card_o = []
        for emb, (ranks, suits, cards) in zip(self.card_embs, card_batches):
            card_o.append(emb(ranks=ranks, suits=suits, cards=cards))

        return torch.cat(card_o, dim=1)


class _CardGroupEmb(nn.Module):

    def __init__(self, env_bldr, dim):
        super().__init__()
        self._env_bldr = env_bldr
        self._dim = dim
        self.rank = nn.Embedding(env_bldr.rules.N_RANKS, dim)
        if self._env_bldr.rules.SUITS_MATTER:
            self.suit = nn.Embedding(env_bldr.rules.N_SUITS, dim)
            self.card = nn.Embedding(env_bldr.rules.N_RANKS * env_bldr.rules.N_SUITS, dim)

    def forward(self, ranks, suits, cards):
        bs, n_cards = cards.shape

        r = ranks.view(-1)
        valid_r = r.ge(0).unsqueeze(1).to(torch.float32)
        r = r.clamp(min=0)
        embs = self.rank(r) * valid_r

        if self._env_bldr.rules.SUITS_MATTER:
            s = suits.view(-1)
            c = cards.view(-1)
            valid_s = s.ge(0).unsqueeze(1).to(torch.float32)
            valid_c = c.ge(0).unsqueeze(1).to(torch.float32)
            s = s.clamp(min=0)
            c = c.clamp(min=0)

            embs += self.card(c) * valid_c + self.suit(s) * valid_s

        return embs.view(bs, n_cards, -1).sum(1)
