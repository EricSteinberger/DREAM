# Copyright (c) 2019 Eric Steinberger


class ActionReservoirBufferBase:

    def __init__(self, env_bldr, max_size, min_prob):
        self._env_bldr = env_bldr
        self._max_size = max_size
        self._min_prob = min_prob

        # track
        self.size = 0
        self.n_entries_seen = 0

    def sample(self, batch_size, device):
        raise NotImplementedError

    def state_dict(self):
        raise NotImplementedError

    def load_state_dict(self, state):
        raise NotImplementedError


class AvgMemorySaverBase:

    def __init__(self, env_bldr, buffer):
        self._env_bldr = env_bldr
        self._buffer = buffer

    def add_step(self, pub_obs, a, legal_actions_mask):
        raise NotImplementedError

    def reset(self, range_idx, sample_weight):
        raise NotImplementedError
