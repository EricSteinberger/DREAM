# Copyright (c) 2019 Eric Steinberger
import copy

import numpy as np
import torch

from NFSP.workers.la.action_buffer._ActionReservoirBufferBase import ActionReservoirBufferBase, AvgMemorySaverBase


class _GameForRes:
    """
    Stores a whole game as one array instead of a copy for each timestep as the naive approach would.

    self.obs_idxs_per_step contains the timesteps (and thus indices in the array) at which the player this memory
    belongs to acted.

    """

    def __init__(self, range_idx):
        self.n_steps_in_game_memory = 0

        self.range_idx = range_idx

        self.obs_sequence = []
        self.obs_idxs_per_step = []

        self.action_buffer = []
        self.legal_actions_mask_buffer = []

    def add(self, pub_obs, a, legal_actions_mask):
        self.obs_idxs_per_step.append(pub_obs.shape[0])
        self.obs_sequence = np.copy(pub_obs)  # NOT append. this is on purpose.
        self.legal_actions_mask_buffer.append(np.copy(legal_actions_mask))
        self.action_buffer.append(a)

        self.n_steps_in_game_memory += 1

    def sample(self):
        idx = np.random.randint(low=0, high=self.n_steps_in_game_memory)
        return {
            "o": self.obs_sequence[:self.obs_idxs_per_step[idx]],
            "mask": self.legal_actions_mask_buffer[idx],
            "a": self.action_buffer[idx],
            "range_idx": self.range_idx
        }


class ActionBufferRNN(ActionReservoirBufferBase):

    def __init__(self, env_bldr, max_size, min_prob):
        super().__init__(env_bldr=env_bldr, max_size=max_size, min_prob=min_prob)

        # stores references to Game subclass objects. One Game instance might be referenced multiple times,
        # depending on the number of steps that it contains. This is to keep equally likely sampling among timesteps.
        self.games = np.array([None for _ in range(self._max_size)], dtype=object)

    def add_game_with_sampling(self, game):
        """ use resevoir sampling """

        for _ in range(game.n_steps_in_game_memory):

            if self.size < self._max_size:
                self.games[self.size] = game
                self.size += 1

            else:
                prob_add = max(float(self._max_size) / float(self.n_entries_seen), self._min_prob)
                if np.random.random() < prob_add:
                    index = np.random.randint(low=0, high=self._max_size)
                    self.games[index] = game

            self.n_entries_seen += 1

    def sample(self, batch_size, device):
        indices = np.random.randint(low=0, high=self.size, size=batch_size)

        samples = [self.games[i].sample() for i in indices]

        batch_legal_action_mask_t = [sample["mask"] for sample in samples]
        batch_legal_action_mask_t = torch.from_numpy(np.array(batch_legal_action_mask_t)).to(device=device)

        batch_action = [sample["a"] for sample in samples]
        batch_action = torch.from_numpy(np.array(batch_action)).to(dtype=torch.long, device=device)

        batch_range_idx = [sample["range_idx"] for sample in samples]
        batch_range_idx = torch.from_numpy(np.array(batch_range_idx)).to(dtype=torch.long, device=device)

        # will be processed to PackedSequence in the NN
        batch_pub_obs = [sample["o"] for sample in samples]

        return batch_pub_obs, \
               batch_action, \
               batch_range_idx, \
               batch_legal_action_mask_t

    def state_dict(self, copy_=False):
        return {
            "games": self.games if not copy else copy.deepcopy(self.games),
            "size": self.size,
            "n_entries_seen": self.n_entries_seen
        }

    def load_state_dict(self, state, copy_=False):
        self.games = state["games"] if not copy_ else copy.deepcopy(state["games"])
        self.size = state["size"]
        self.n_entries_seen = state["n_entries_seen"]


class AvgMemorySaverRNN(AvgMemorySaverBase):

    def __init__(self, env_bldr, buffer):
        super().__init__(env_bldr=env_bldr, buffer=buffer)
        self._game_memory = None

    def add_step(self, pub_obs, a, legal_actions_mask):
        self._game_memory.add(pub_obs=pub_obs, a=a, legal_actions_mask=legal_actions_mask)

    def reset(self, range_idx):
        """ Call with env reset """
        if self._game_memory is not None:
            self._buffer.add_game_with_sampling(self._game_memory)
        self._game_memory = _GameForRes(range_idx=range_idx)
