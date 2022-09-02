#---------------------------------------------------------------------------------------------------#
# File name: environment.py                                                                         #
# Autor: Chrissi2802                                                                                #
# Created on: 21.07.2022                                                                            #
#---------------------------------------------------------------------------------------------------#
# Reinforcement Learning - BlackJack Player
# Self-learning BackJack player based on Reinforcement Learning methods
# Exact description in the functions.
# This file provides the environment.


import gym
from gym import spaces
from typing import Optional
from deck import Deck


# 1 = Ace, 2-10 = Number cards, Jack / Queen / King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def cmp(score_player, score_dealer):
    """This function compares the player's cards with those of the dealer."""
    return float(score_player > score_dealer) - float(score_player < score_dealer)


def draw_card(np_random):
    """This function draws a random card."""
    return int(np_random.choice(deck))


def draw_hand(np_random):
    """This function makes a hand from two random cards."""
    return [draw_card(np_random), draw_card(np_random)]


def usable_ace(hand):
    """This function returns whether there is a usable ace on the hand."""
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  
    """This function returns the sum of the current hand."""
    if (usable_ace(hand)):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):
    """This function checks whether the hand is a bust."""
    return sum_hand(hand) > 21


def score(hand):  
    """This function determines the current score of the hand, zero means bust."""
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):
    """This function determines if the hand is a natural blackjack."""
    return sorted(hand) == [1, 10]


def can_double_down(hand, actionstaken):
    """This function checks whether double down can be played."""
    return len(hand) == 2 and actionstaken == 0


def can_split(hand, actionstaken):
    """This function checks whether splitting is possible."""
    return can_double_down(hand, actionstaken) and hand[0] == hand[1]


class Blackjack_Env(gym.Env):
    """
    Blackjack is a card game where the goal is to beat the dealer by obtaining cards
    that sum to closer to 21 (without going over 21) than the dealers cards.

    ### Description

        Card Values:
            - Face cards (Jack, Queen, King) have a point value of 10.
            - Aces can either count as 11 (called a "usable ace") or 1.
            - Numerical cards (2-9) have a value equal to their number.

        This game is played with an infinite deck or with replacement.
        The game starts with the dealer having one face up and one face down card,
        while the player has two face up cards.

    ### Action Space
    There are four actions: stick (0), hit (1), double down (2), split(3)

    This environment corresponds to the version of the blackjack problem
    described in Richard S. Sutton and Andrew G. Barto. 
    Reinforcement Learning: An Introduction. MIT Press, Cambridge, MA, 2 edition, 2018.

    ### Rewards
        - win game: +1
        - lose game: -1
        - draw game: 0
        - win game with natural blackjack: +1.5
    """

    def __init__(self, natural = False, sab = False):
        """Initialisation of the Blackjack_Env class (constructor).
            Input:
            natural
            sab
        """

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(32), spaces.Discrete(11),
                                               spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2)))
        self.natural = natural
        self.sab = sab
        self.actionstaken = 0

    def step(self, action):
        """This method performs a learning step.
            Input:
            action
        """

        assert self.action_space.contains(action)
        
        if (action == 0):  
            # stick: play out the dealers hand, and score

            terminated = True

            while (sum_hand(self.dealer) < 17): # 16, 18
                self.dealer.append(draw_card(self.np_random))

            reward = cmp(score(self.player), score(self.dealer))

            
            if (self.sab and is_natural(self.player) and not is_natural(self.dealer)):
                # Player automatically wins
                reward = 1.0
                #reward = 1.0 * 2.0

            elif (not self.sab and self.natural and is_natural(self.player) and reward == 1.0):
                # Natural gives extra points, but doesn't autowin. Legacy implementation
                reward = 1.5
                #reward = 1.5 * 2.0
            
            self.actionstaken += 1  

        elif (action == 1):  
            # hit: add a card to players hand and return

            self.player.append(draw_card(self.np_random))

            if (is_bust(self.player)):
                terminated = True
                reward = -1.0
                #reward = -1.0 * 2.0
            else:
                terminated = False
                reward = 0.0

            self.actionstaken += 1  

        elif (action == 2):
            # double down

            self.player.append(draw_card(self.np_random))

            if (is_bust(self.player)):
                terminated = True
                reward = -2.0
                #reward = -2.0 * 2.0
            else:
                terminated = False

                while (sum_hand(self.dealer) < 17): # 16, 18
                    self.dealer.append(draw_card(self.np_random))

                reward = 2.0 * cmp(score(self.player), score(self.dealer))
                #reward = 2.0 * cmp(score(self.player), score(self.dealer)) * 2.0

            self.actionstaken += 1  

        elif (action == 3):
            # split
            # ToDo:
            # Splitting does not work correctly
            
            self.player.append(draw_card(self.np_random))

            if (is_bust(self.player)):
                terminated = True
                reward = -1.0
                #reward = -1.0 * 2.0
            else:
                terminated = False
                reward = 0.0

            self.actionstaken += 1 
        
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        """This method gets the observations."""

        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player), can_double_down(self.player, self.actionstaken), can_split(self.player, self.actionstaken))

    def reset(self, seed: Optional[int] = None, return_info: bool = False):
        """This method resets the environment.
            Input:
            seed
            return_info
        """

        super().reset(seed = seed)

        self.dealer = draw_hand(self.np_random)
        self.player = draw_hand(self.np_random)
        self.actionstaken = 0

        if (not return_info):
            return self._get_obs()
        else:
            return self._get_obs(), {}


class Blackjack_Env_CC(gym.Env):
    """
    Blackjack is a card game where the goal is to beat the dealer by obtaining cards
    that sum to closer to 21 (without going over 21) than the dealers cards.

    ### Description

        Card Values:
            - Face cards (Jack, Queen, King) have a point value of 10.
            - Aces can either count as 11 (called a "usable ace") or 1.
            - Numerical cards (2-9) have a value equal to their number.

        The game starts with the dealer having one face up and one face down card,
        while the player has two face up cards.

    ### Action Space
    There are four actions: stick (0), hit (1), double down (2), split(3)

    This environment corresponds to the version of the blackjack problem
    described in Richard S. Sutton and Andrew G. Barto. 
    Reinforcement Learning: An Introduction. MIT Press, Cambridge, MA, 2 edition, 2018.

    ### Rewards
        - win game: +1
        - lose game: -1
        - draw game: 0
        - win game with natural blackjack: +1.5

    ### A special feature here is that card counting is possible.
    """

    def __init__(self, natural = False, sab = False):
        """Initialisation of the Blackjack_Env_CC class (constructor).
            Input:
            natural
            sab
        """

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Tuple((spaces.Discrete(32), spaces.Discrete(11),
                                               spaces.Discrete(2), spaces.Discrete(2), spaces.Discrete(2)))
        self.natural = natural
        self.sab = sab
        self.actionstaken = 0
        self.deck = Deck(seed = 0, number_of_decks = 4)

    def step(self, action):
        """This method performs a learning step.
            Input:
            action
        """

        assert self.action_space.contains(action)
        
        if (action == 0):  
            # stick: play out the dealers hand, and score

            terminated = True

            while (sum_hand(self.dealer) < 17): # 16, 18
                self.dealer.append(self.deck.draw_card())

            reward = cmp(score(self.player), score(self.dealer))

            
            if (self.sab and is_natural(self.player) and not is_natural(self.dealer)):
                # Player automatically wins
                reward = 1.0
                #reward = 1.0 * 2.0

            elif (not self.sab and self.natural and is_natural(self.player) and reward == 1.0):
                # Natural gives extra points, but doesn't autowin. Legacy implementation
                reward = 1.5
                #reward = 1.5 * 2.0
            
            self.actionstaken += 1  

        elif (action == 1):  
            # hit: add a card to players hand and return

            self.player.append(self.deck.draw_card())

            if (is_bust(self.player)):
                terminated = True
                reward = -1.0
                #reward = -1.0 * 2.0
            else:
                terminated = False
                reward = 0.0

            self.actionstaken += 1  

        elif (action == 2):
            # double down

            self.player.append(self.deck.draw_card())

            if (is_bust(self.player)):
                terminated = True
                reward = -2.0
                #reward = -2.0 * 2.0
            else:
                terminated = False

                while (sum_hand(self.dealer) < 17): # 16, 18
                    self.dealer.append(self.deck.draw_card())

                reward = 2.0 * cmp(score(self.player), score(self.dealer))
                #reward = 2.0 * cmp(score(self.player), score(self.dealer)) * 2.0

            self.actionstaken += 1  

        elif (action == 3):
            # split
            # ToDo:
            # Splitting does not work correctly
            
            self.player.append(self.deck.draw_card())

            if (is_bust(self.player)):
                terminated = True
                reward = -1.0
                #reward = -1.0 * 2.0
            else:
                terminated = False
                reward = 0.0

            self.actionstaken += 1 
        
        return self._get_obs(), reward, terminated, False, {}

    def _get_obs(self):
        """This method gets the observations."""

        return (sum_hand(self.player), self.dealer[0], usable_ace(self.player), can_double_down(self.player, self.actionstaken), can_split(self.player, self.actionstaken), self.deck.total_points, self.deck.unseen_cards)

    def reset(self, seed: Optional[int] = None, return_info: bool = False):
        """This method resets the environment.
            Input:
            seed
            return_info
        """

        super().reset(seed = seed)

        self.deck.init_deck()
        self.dealer = self.deck.draw_hand()
        self.player = self.deck.draw_hand()
        self.actionstaken = 0

        if (not return_info):
            return self._get_obs()
        else:
            return self._get_obs(), {}

