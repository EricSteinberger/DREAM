# Copyright (c) 2019 Eric Steinberger


import numpy as np

from PokerRL.game.PokerEnvStateDictEnums import PlayerDictIdxs


class PokerPlayer:
    """
    Holds information about the player state in a PokerEnv and gives an interface to modify player variables.
    """

    def __init__(self, seat_id, poker_env, is_evaluating, starting_stack, stack_randomization_range):
        assert isinstance(seat_id, int)
        assert isinstance(starting_stack, int)
        assert isinstance(stack_randomization_range, tuple)
        assert len(stack_randomization_range) == 2

        self.seat_id = seat_id
        self.poker_env = poker_env
        self._base_starting_stack = starting_stack
        self.stack_randomization_range = stack_randomization_range

        self.IS_EVALUATING = is_evaluating

        # player vars
        self.hand = None  # np.ndarray
        self.hand_rank = None
        self.current_bet = None
        self.stack = None
        self.starting_stack_this_episode = None

        # flags for table management
        self.is_allin = None
        self.folded_this_episode = None
        self.has_acted_this_round = None

        # each player has his own side-pot rank.
        # rank0 gets only the main_pot. rank1 gets main-pot and self.side_pots[1], ...
        self.side_pot_rank = None

        # sets all the above
        self.reset()

    def reset(self):

        if self.IS_EVALUATING or (self.stack_randomization_range[0] == 0 and self.stack_randomization_range[1] == 0):
            self.starting_stack_this_episode = self._base_starting_stack
        else:
            self.starting_stack_this_episode = \
                max(
                    self.poker_env.BIG_BLIND,
                    np.random.randint(
                        low=self._base_starting_stack - np.abs(self.stack_randomization_range[0]),
                        high=self._base_starting_stack + self.stack_randomization_range[1] + 1)
                )

        # episode info
        self.stack = self.starting_stack_this_episode
        self.hand = []  # [[rank, suit], ...]
        self.current_bet = 0

        # flags for table management
        self.is_allin = False
        self.folded_this_episode = False
        self.has_acted_this_round = False
        self.side_pot_rank = -1

    def bet_raise(self, total_bet_amount):
        """
        assumes bet is 100% legal and bet_size <= stack size

        Args:
            total_bet_amount (int)
        """

        self.has_acted_this_round = True
        self.stack -= (total_bet_amount - self.current_bet)
        self.current_bet = total_bet_amount

        if self.stack == 0:
            self.is_allin = True

    def check_call(self, total_to_call):
        """
        Assumes having total_to_call as self.current_bet is 100% legal and just does it.

        Args:
            total_to_call (int)
        """
        self.has_acted_this_round = True
        delta = int(total_to_call - self.current_bet)
        self.stack -= delta
        self.current_bet = int(total_to_call)  # equal to += delta but is faster

        if self.stack == 0:
            self.is_allin = True

    def fold(self):
        self.has_acted_this_round = True
        self.folded_this_episode = True

    def award(self, amount):
        self.stack += amount

    def refund_from_bet(self, amount):
        self.stack += amount
        self.current_bet -= amount

    def player_state(self):
        return (self.seat_id,
                self.stack,
                self.folded_this_episode,
                self.is_allin)

    def state_dict(self):
        return {
            PlayerDictIdxs.seat_id: self.seat_id,  # int
            PlayerDictIdxs.hand: np.copy(self.hand),  # np array
            PlayerDictIdxs.hand_rank: self.hand_rank,  # int by value
            PlayerDictIdxs.stack: self.stack,  # int by value
            PlayerDictIdxs.current_bet: self.current_bet,  # int by value
            PlayerDictIdxs.is_allin: self.is_allin,  # bool by value
            PlayerDictIdxs.folded_this_episode: self.folded_this_episode,  # bool by value
            PlayerDictIdxs.has_acted_this_round: self.has_acted_this_round,  # bool by value
            PlayerDictIdxs.side_pot_rank: self.side_pot_rank,  # int by value
            PlayerDictIdxs.starting_stack_this_episode: self.starting_stack_this_episode  # int by value
        }

    def load_state_dict(self, state, blank_private_info):
        self.seat_id = state[PlayerDictIdxs.seat_id]
        self.stack = state[PlayerDictIdxs.stack]
        self.starting_stack_this_episode = state[PlayerDictIdxs.starting_stack_this_episode]
        self.current_bet = state[PlayerDictIdxs.current_bet]
        self.is_allin = state[PlayerDictIdxs.is_allin]
        self.folded_this_episode = state[PlayerDictIdxs.folded_this_episode]
        self.has_acted_this_round = state[PlayerDictIdxs.has_acted_this_round]
        self.side_pot_rank = state[PlayerDictIdxs.side_pot_rank]

        if blank_private_info:
            self.hand = None
            self.hand_rank = None
        else:
            self.hand = np.copy(state[PlayerDictIdxs.hand])
            self.hand_rank = state[PlayerDictIdxs.hand_rank]
