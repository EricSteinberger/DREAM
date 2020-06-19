# Copyright (c) 2019 Eric Steinberger


import torch
import torch.nn.functional as nnf

from PokerRL.rl import rl_util
from PokerRL.rl.neural.AvrgStrategyNet import AvrgStrategyNet
from PokerRL.rl.neural.NetWrapperBase import NetWrapperArgsBase as _NetWrapperArgsBase
from PokerRL.rl.neural.NetWrapperBase import NetWrapperBase as _NetWrapperBase


class AvgWrapper(_NetWrapperBase):

    def __init__(self, owner, env_bldr, avg_training_args):
        super().__init__(
            net=AvrgStrategyNet(avrg_net_args=avg_training_args.avg_net_args, env_bldr=env_bldr,
                                device=avg_training_args.device_training),
            env_bldr=env_bldr,
            args=avg_training_args,
            owner=owner,
            device=avg_training_args.device_training,
        )
        self._all_range_idxs = torch.arange(self._env_bldr.rules.RANGE_SIZE, device=self.device, dtype=torch.long)

    def get_a_probs(self, pub_obses, range_idxs, legal_actions_lists):
        """
        Args:
            pub_obses (list):             list of np arrays of shape [np.arr([history_len, n_features]), ...)
            range_idxs (np.ndarray):    array of range_idxs (one for each pub_obs) tensor([2, 421, 58, 912, ...])
            legal_actions_lists (list:  list of lists. each 2nd level lists contains ints representing legal actions
        """
        with torch.no_grad():
            masks = rl_util.batch_get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                              legal_actions_lists=legal_actions_lists,
                                                              device=self.device)

            return self.get_a_probs2(pub_obses=pub_obses, range_idxs=range_idxs, legal_action_masks=masks)

    def get_a_probs2(self, pub_obses, range_idxs, legal_action_masks):
        with torch.no_grad():
            pred = self._net(pub_obses=pub_obses,
                             range_idxs=torch.from_numpy(range_idxs).to(dtype=torch.long, device=self.device),
                             legal_action_masks=legal_action_masks)

            return nnf.softmax(pred, dim=-1).cpu().numpy()

    def get_a_probs_for_each_hand(self, pub_obs, legal_actions_list):
        assert isinstance(legal_actions_list[0], int), "all hands can do the same actions. no need to batch"

        with torch.no_grad():
            mask = rl_util.get_legal_action_mask_torch(n_actions=self._env_bldr.N_ACTIONS,
                                                       legal_actions_list=legal_actions_list,
                                                       device=self.device, dtype=torch.uint8)
            mask = mask.unsqueeze(0).expand(self._env_bldr.rules.RANGE_SIZE, -1)

            pred = self._net(pub_obses=[pub_obs] * self._env_bldr.rules.RANGE_SIZE,
                             range_idxs=self._all_range_idxs,
                             legal_action_masks=mask)

            return nnf.softmax(pred, dim=1).cpu().numpy()

    def _mini_batch_loop(self, buffer, grad_mngr):
        batch_pub_obs_t, \
        batch_a_t, \
        batch_range_idx, \
        batch_weights, \
        batch_legal_action_mask_t \
            = buffer.sample(device=self.device, batch_size=self._args.batch_size)

        # [batch_size, n_actions]
        pred = self._net(pub_obses=batch_pub_obs_t,
                         range_idxs=batch_range_idx,
                         legal_action_masks=batch_legal_action_mask_t)

        grad_mngr.backprop(pred=pred, target=batch_a_t, loss_weights=batch_weights)


class AvgWrapperArgs(_NetWrapperArgsBase):

    def __init__(self,
                 avg_net_args,
                 res_buf_size=1e6,
                 min_prob_add_res_buf=0.0,
                 batch_size=512,
                 loss_str="ce",
                 optim_str="rms",
                 lr=0.0002,
                 device_training="cpu",
                 grad_norm_clipping=10.0,
                 ):
        super().__init__(batch_size=batch_size,
                         optim_str=optim_str,
                         loss_str=loss_str,
                         lr=lr,
                         grad_norm_clipping=grad_norm_clipping,
                         device_training=device_training)
        self.avg_net_args = avg_net_args
        self.res_buf_size = int(res_buf_size)
        self.min_prob_res_buf = min_prob_add_res_buf
