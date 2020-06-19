# Copyright (c) 2019 Eric Steinberger


import numpy as np
import torch

from NFSP.workers.la.action_buffer._ActionReservoirBufferBase import ActionReservoirBufferBase, AvgMemorySaverBase


class ActionBufferFLAT(ActionReservoirBufferBase):

    def __init__(self, env_bldr, max_size, min_prob):
        super().__init__(env_bldr=env_bldr, max_size=max_size, min_prob=min_prob)

        self.storage_device = torch.device("cpu")

        self._pub_obs_buffer = torch.empty(size=(self._max_size, self._env_bldr.pub_obs_size), dtype=torch.float32,
                                           device=self.storage_device)
        self._action_buffer = torch.empty(size=(self._max_size,), dtype=torch.long, device=self.storage_device)
        self._range_idx_buffer = torch.empty(size=(self._max_size,), dtype=torch.long, device=self.storage_device)
        self._sample_weight_buffer = torch.empty(size=(self._max_size,), dtype=torch.float32,
                                                 device=self.storage_device)
        self._legal_action_mask_buffer = torch.empty(size=(self._max_size, self._env_bldr.N_ACTIONS),
                                                     dtype=torch.float32,
                                                     device=self.storage_device)

        self._highest_sample_weight = 1

    def add_step_with_sampling(self, pub_obs, a, legal_actions_mask, range_idx, sample_weight):
        """ use resevoir sampling """

        if self.size < self._max_size:
            self._insert(idx=self.size,
                         pub_obs=pub_obs, a=a, legal_actions_mask=legal_actions_mask, range_idx=range_idx,
                         sample_weight=sample_weight)
            self.size += 1

        else:
            prob_add = max(float(self._max_size) / float(self.n_entries_seen), self._min_prob)
            if np.random.random() < prob_add:
                self._insert(idx=np.random.randint(low=0, high=self._max_size),
                             pub_obs=pub_obs, a=a, legal_actions_mask=legal_actions_mask, range_idx=range_idx,
                             sample_weight=sample_weight)

        self.n_entries_seen += 1

    def _insert(self, idx, pub_obs, a, sample_weight, legal_actions_mask, range_idx):
        self._pub_obs_buffer[idx] = torch.from_numpy(pub_obs).to(self.storage_device)
        self._action_buffer[idx] = a
        self._range_idx_buffer[idx] = range_idx
        self._sample_weight_buffer[idx] = sample_weight
        self._legal_action_mask_buffer[idx] = torch.from_numpy(legal_actions_mask).to(self.storage_device)

        self._highest_sample_weight = max(self._highest_sample_weight, sample_weight)

    def sample(self, batch_size, device):
        indices = torch.randint(0, self.size, (batch_size,), dtype=torch.long, device=device)

        return self._pub_obs_buffer[indices].to(device), \
               self._action_buffer[indices].to(device), \
               self._range_idx_buffer[indices].to(device), \
               self._sample_weight_buffer[indices].to(device) / self._highest_sample_weight, \
               self._legal_action_mask_buffer[indices].to(device)

    def state_dict(self, copy_=False):
        if copy_:
            return {
                "pub_obs_buffer": self._pub_obs_buffer.cpu().clone(),
                "action_buffer": self._action_buffer.cpu().clone(),
                "range_idx_buffer": self._range_idx_buffer.cpu().clone(),
                "legal_action_mask_buffer": self._legal_action_mask_buffer.cpu().clone(),
                "sample_weight_buffer": self._sample_weight_buffer.cpu().clone(),
                "size": self.size,
                "n_entries_seen": self.n_entries_seen
            }

        return {
            "pub_obs_buffer": self._pub_obs_buffer.cpu(),
            "action_buffer": self._action_buffer.cpu(),
            "range_idx_buffer": self._range_idx_buffer.cpu(),
            "legal_action_mask_buffer": self._legal_action_mask_buffer.cpu(),
            "sample_weight_buffer": self._sample_weight_buffer,
            "size": self.size,
            "n_entries_seen": self.n_entries_seen
        }

    def load_state_dict(self, state, copy_=False):
        if copy_:
            self._pub_obs_buffer = state["pub_obs_buffer"].clone().to(self.storage_device)
            self._action_buffer = state["action_buffer"].clone().to(self.storage_device)
            self._range_idx_buffer = state["range_idx_buffer"].clone().to(self.storage_device)
            self._sample_weight_buffer = state["sample_weight_buffer"].clone().to(self.storage_device)
            self._legal_action_mask_buffer = state["legal_action_mask_buffer"].clone().to(self.storage_device)
            self.size = state["size"]
            self.n_entries_seen = state["n_entries_seen"]
        else:
            self._pub_obs_buffer = state["pub_obs_buffer"].to(self.storage_device)
            self._action_buffer = state["action_buffer"].to(self.storage_device)
            self._range_idx_buffer = state["range_idx_buffer"].to(self.storage_device)
            self._sample_weight_buffer = state["sample_weight_buffer"].to(self.storage_device)
            self._legal_action_mask_buffer = state["legal_action_mask_buffer"].to(self.storage_device)
            self.size = state["size"]
            self.n_entries_seen = state["n_entries_seen"]


class AvgMemorySaverFLAT(AvgMemorySaverBase):

    def __init__(self, env_bldr, buffer):
        super().__init__(env_bldr=env_bldr, buffer=buffer, )
        self._range_idx = None
        self.sample_weight = None

    def add_step(self, pub_obs, a, legal_actions_mask):
        self._buffer.add_step_with_sampling(pub_obs=pub_obs,
                                            range_idx=self._range_idx,
                                            sample_weight=self.sample_weight,
                                            a=a,
                                            legal_actions_mask=legal_actions_mask)

    def reset(self, range_idx, sample_weight):
        self._range_idx = range_idx
        self.sample_weight = sample_weight
