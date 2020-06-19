# Copyright (c) 2019 Eric Steinberger


import torch

from PokerRL.rl import rl_util


class NetWrapperBase:

    def __init__(self, net, env_bldr, args, owner, device, train_before_enough_samples_in_buf=False):
        self._env_bldr = env_bldr
        self._args = args
        self.owner = owner
        self.device = device
        self._train_before_enough_samples_in_buf = train_before_enough_samples_in_buf

        self._criterion = rl_util.str_to_loss_cls(self._args.loss_str)
        self.loss_last_batch = None

        self._net = net
        self.eval()

    def get_grads_one_batch_from_buffer(self, buffer):
        if (not self._train_before_enough_samples_in_buf) and (buffer.size < self._args.batch_size):
            return

        self.train()
        # Might want to do multiple mini batches because all together don't fit into every gpu's vRAM.
        _grad_mngr = _GradManager(net=self._net, args=self._args, criterion=self._criterion)
        self._mini_batch_loop(buffer=buffer, grad_mngr=_grad_mngr)

        self.loss_last_batch = _grad_mngr.get_loss_sum()
        self.eval()

        return _grad_mngr.average()

    def _mini_batch_loop(self, buffer, grad_mngr):
        raise NotImplementedError

    def load_net_state_dict(self, state_dict):
        self._net.load_state_dict(state_dict)

    def net_state_dict(self):
        return self._net.state_dict()

    def train(self):
        self._net.train()

    def eval(self):
        self._net.eval()

    def state_dict(self):
        """ Override, if necessary"""
        return self.net_state_dict()

    def load_state_dict(self, state):
        """ Override, if necessary"""
        self.load_net_state_dict(state)


class NetWrapperArgsBase:

    def __init__(self,
                 batch_size,
                 optim_str,
                 loss_str,
                 lr,
                 grad_norm_clipping,
                 device_training
                 ):
        assert isinstance(device_training, str), "Please pass a string (either 'cpu' or 'cuda')!"
        self.batch_size = batch_size
        self.optim_str = optim_str
        self.loss_str = loss_str
        self.lr = lr
        self.grad_norm_clipping = grad_norm_clipping
        self.device_training = torch.device(device_training)


class _GradManager:

    def __init__(self, args, net, criterion):
        self.net = net
        self.args = args
        self.criterion = criterion
        self._grads = {}
        self._loss_sum = 0.0
        for name, _ in net.named_parameters():
            self._grads[name] = []

    def backprop(self, pred, target, loss_weights=None):
        self.net.zero_grad()
        if loss_weights is None:
            loss = self.criterion(pred, target)
        else:
            loss = self.criterion(pred, target, loss_weights)
        loss.backward()
        self._loss_sum += loss.item()
        self._add()

    def backprop_from_loss(self, loss):
        self.net.zero_grad()
        loss.backward()
        self._loss_sum += loss.item()
        self._add()

    def _add(self):
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                self._grads[name].append(param.grad.data.clone())

    def average(self):
        for name, param in self.net.named_parameters():
            if param.grad is not None:
                self._grads[name] = self._grads[name][0]
        return self._grads

    def get_loss_sum(self):
        return self._loss_sum
