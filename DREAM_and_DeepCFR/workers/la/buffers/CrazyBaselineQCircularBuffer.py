# Copyright (c) Eric Steinberger 2020

import numpy as np
import torch


class CrazyBaselineQCircularBuffer:
    """
    Circular buffer compatible with all NN architectures
    """

    def __init__(self, owner, max_size, env_bldr, nn_type):
        self._owner = owner
        self._env_bldr = env_bldr
        self._max_size = int(max_size)

        self._nn_type = nn_type
        self.device = torch.device("cpu")
        self.size = 0

        if nn_type == "recurrent":
            self._pub_obs_buffer = np.empty(shape=(max_size,), dtype=object)
        elif nn_type == "feedforward":
            self._pub_obs_buffer = torch.zeros((max_size, self._env_bldr.pub_obs_size), dtype=torch.float32,
                                               device=self.device)
        else:
            raise ValueError(nn_type)

        self._range_idx_buffer = torch.zeros((max_size,), dtype=torch.long, device=self.device)
        self._legal_action_mask_buffer = torch.zeros((max_size, env_bldr.N_ACTIONS,),
                                                     dtype=torch.float32, device=self.device)

        self._top = None

        self._a_buffer = None
        self._strat_tp1_buffer = None
        self._r_buffer = None
        self._done = None
        self._pub_obs_buffer_tp1 = None
        self._legal_action_mask_buffer_tp1 = None
        self.reset()

    def _np_to_torch(self, arr):
        return torch.from_numpy(np.copy(arr)).to(self.device)

    def _random_idx(self):
        return np.random.randint(low=0, high=self._max_size)

    def add(self, pub_obs, range_idx_crazy_embedded, legal_action_mask, r, a, done, legal_action_mask_tp1, pub_obs_tp1,
            strat_tp1):
        if self._nn_type == "feedforward":
            pub_obs = torch.from_numpy(pub_obs)
            pub_obs_tp1 = torch.from_numpy(pub_obs_tp1)

        self._pub_obs_buffer[self._top] = pub_obs
        self._pub_obs_buffer_tp1[self._top] = pub_obs_tp1

        self._range_idx_buffer[self._top] = range_idx_crazy_embedded

        self._legal_action_mask_buffer[self._top] = legal_action_mask
        self._legal_action_mask_buffer_tp1[self._top] = legal_action_mask_tp1

        self._r_buffer[self._top] = r
        self._a_buffer[self._top] = a
        self._done[self._top] = float(done)

        self._strat_tp1_buffer[self._top] = strat_tp1
        if self.size < self._max_size:
            self.size += 1

        self._top = (self._top + 1) % self._max_size

    def sample(self, batch_size, device):
        indices = torch.randint(0, self.size, (batch_size,), dtype=torch.long, device=self.device)

        if self._nn_type == "recurrent":
            obses = self._pub_obs_buffer[indices.cpu().numpy()]
            obses_tp1 = self._pub_obs_buffer_tp1[indices.cpu().numpy()]
        elif self._nn_type == "feedforward":
            obses = self._pub_obs_buffer[indices].to(device)
            obses_tp1 = self._pub_obs_buffer_tp1[indices].to(device)
        else:
            raise NotImplementedError

        return \
            obses, \
            self._range_idx_buffer[indices].to(device), \
            self._legal_action_mask_buffer[indices].to(device), \
            self._a_buffer[indices].to(device), \
            self._r_buffer[indices].to(device), \
            obses_tp1, \
            self._legal_action_mask_buffer_tp1[indices].to(device), \
            self._done[indices].to(device), \
            self._strat_tp1_buffer[indices].to(device)

    def reset(self):
        self._top = 0
        self.size = 0

        if self._nn_type == "recurrent":
            self._pub_obs_buffer = np.empty(shape=(self._max_size,), dtype=object)
        elif self._nn_type == "feedforward":
            self._pub_obs_buffer = torch.zeros((self._max_size, self._env_bldr.pub_obs_size), dtype=torch.float32,
                                               device=self.device)
        else:
            raise ValueError(self._nn_type)

        self._range_idx_buffer = torch.zeros((self._max_size,), dtype=torch.long, device=self.device)
        self._legal_action_mask_buffer = torch.zeros((self._max_size, self._env_bldr.N_ACTIONS,),
                                                     dtype=torch.float32, device=self.device)

        self._a_buffer = torch.zeros((self._max_size,), dtype=torch.long, device=self.device)
        self._strat_tp1_buffer = torch.zeros((self._max_size, self._env_bldr.N_ACTIONS), dtype=torch.float32,
                                             device=self.device)
        self._r_buffer = torch.zeros((self._max_size,), dtype=torch.float32, device=self.device)
        self._done = torch.zeros((self._max_size,), dtype=torch.float32, device=self.device)

        if self._nn_type == "recurrent":
            self._pub_obs_buffer_tp1 = np.empty(shape=(self._max_size,), dtype=object)
        elif self._nn_type == "feedforward":
            self._pub_obs_buffer_tp1 = torch.zeros((self._max_size, self._env_bldr.pub_obs_size), dtype=torch.float32,
                                                   device=self.device)
        else:
            raise ValueError(self._nn_type)

        self._legal_action_mask_buffer_tp1 = torch.zeros((self._max_size, self._env_bldr.N_ACTIONS,),
                                                         dtype=torch.uint8, device=self.device)

    def state_dict(self):
        return {
            "owner": self._owner,
            "max_size": self._max_size,
            "nn_type": self._nn_type,
            "size": self.size,
            "pub_obs_buffer": self._pub_obs_buffer,
            "range_idx_buffer": self._range_idx_buffer,
            "legal_action_mask_buffer": self._legal_action_mask_buffer,
            "a": self._a_buffer,
            "q": self._r_buffer,
            "legal_action_mask_buffer_tp1": self._legal_action_mask_buffer_tp1,
            "pub_obs_buffer_tp1": self._pub_obs_buffer_tp1,
            "done": self._done,
            "strat_tp1": self._strat_tp1_buffer,
        }

    def load_state_dict(self, state):
        assert self._owner == state["owner"]
        assert self._max_size == state["max_size"]
        assert self._nn_type == state["nn_type"]

        self.size = state["size"]

        if self._nn_type == "recurrent":
            self._pub_obs_buffer = state["pub_obs_buffer"]
            self._range_idx_buffer = state["range_idx_buffer"]
            self._legal_action_mask_buffer = state["legal_action_mask_buffer"]

        elif self._nn_type == "feedforward":
            self._pub_obs_buffer = state["pub_obs_buffer"].to(self.device)
            self._range_idx_buffer = state["range_idx_buffer"].to(self.device)
            self._legal_action_mask_buffer = state["legal_action_mask_buffer"].to(self.device)

        else:
            raise ValueError(self._nn_type)

        self._a_buffer = state["a"]
        self._r_buffer = state["q"]
        self._legal_action_mask_buffer_tp1 = state["legal_action_mask_buffer_tp1"]
        self._done = state["done"]
        self._pub_obs_buffer_tp1 = state["pub_obs_buffer_tp1"]
        self._strat_tp1_buffer = state["strat_tp1"]
