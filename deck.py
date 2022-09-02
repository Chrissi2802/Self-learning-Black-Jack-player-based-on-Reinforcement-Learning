#---------------------------------------------------------------------------------------------------#
# File name: deck.py                                                                                #
# Autor: Chrissi2802                                                                                #
# Created on: 21.07.2022                                                                            #
#---------------------------------------------------------------------------------------------------#
# Reinforcement Learning - BlackJack Player
# Self-learning BackJack player based on Reinforcement Learning methods
# Exact description in the functions.
# This file provides the deck.


import random


class Deck():
    """This class provides a deck of cards."""

    def __init__(self, seed = 0, number_of_decks = 6, low_limit = 6, high_limit = 10) -> None:
        """Initialisation of the Deck class (constructor).
            Input:
            seed: seed for the random actions, integer
            number_of_decks: Lower limit for card counting, integer
            low_limit: Lower limit for card counting, integer
            high_limit: Upper limit for card counting, integer
        """

        self.random = random.Random(seed)
        self.number_of_decks = number_of_decks
        self.low_limit = low_limit
        self.high_limit = high_limit
        self.init_deck()

    def init_deck(self):
        """This method initialises a deck of cards."""

        # 1 = Ace, 2-10 = Number cards, Jack / Queen / King = 10
        self.deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * self.number_of_decks
        self.random.shuffle(self.deck)
        self.unseen_cards = len(self.deck)
        self.total_points = 0
        self.card_counter = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0}

    def draw_card(self):
        """This methode draws a random card and makes card counting possible."""

        card = self.deck.pop(0)
        self.unseen_cards -= 1

        if (card >= self.high_limit or card == 1):
            # normally: 10, Jack, Queen, King, Ace
            self.total_points += 1

        elif (card <= self.low_limit):
            # normally: 2 - 6
            self.total_points -= 1

        self.card_counter[card] += 1

        return card
    
    def draw_hand(self):
        """This methde makes a hand from two random cards."""

        return [self.draw_card(), self.draw_card()]

