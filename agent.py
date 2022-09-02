#---------------------------------------------------------------------------------------------------#
# File name: agent.py                                                                               #
# Autor: Chrissi2802                                                                                #
# Created on: 21.07.2022                                                                            #
#---------------------------------------------------------------------------------------------------#
# Reinforcement Learning - BlackJack Player
# Self-learning BackJack player based on Reinforcement Learning methods
# Exact description in the functions.
# This file provides the agents.


import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict


class Agent_Q():
    """This class provides an agent that works according to Q-learning."""

    def __init__(self, env, epsilon = 1.0, learning_rate = 0.5, gamma = 0.9, epochs = 50000):
        """Initialisation of the Agent_Q class (constructor).
            Input:
            env: Blackjack Environment
            epsilon: Probability of selecting random action instead of the optimal action
            learning_rate: Learning Rate
            gamma: Gamma discount factor
            epochs: Number of epochs to be trained
        """

        self.env = env
        self.valid_actions = list(range(self.env.action_space.n))

        # Set parameters of the learning agent
        self.Q = dict()                     # Q-table
        self.epsilon = epsilon              # Random exploration factor
        self.learning_rate = learning_rate  # Learning rate
        self.gamma = gamma                  # Discount factor
        self.epochs = epochs                # Epochs
        self.epochs_left = epochs
        self.small_decrement = (0.1 * epsilon) / (0.3 * epochs) # reduces epsilon slowly
        self.big_decrement = (0.8 * epsilon) / (0.4 * epochs)   # reduces epilon faster

    def update_parameters(self):
        """This method updates epsilon and learning_rate after each action and sets them to 0 when there is no more learning."""

        # epsilon will reduce linearly until it reaches 0 based on epochs
        if (self.epochs_left > 0):
            self.epsilon -= self.small_decrement
        else:
            self.epsilon = 0.0
            self.learning_rate = 0.0

        self.epochs_left -= 1

    def create_Q_if_new_observation(self, observation):
        """This method sets the initial Q-values to 0.0 if the observation is not already included in the Q-table.
            Input:
            obseration
        """

        if (observation not in self.Q):
            self.Q[observation] = dict((action, 0.0) for action in self.valid_actions)

    def get_maxQ(self, observation):
        """This method is called when the agent is asked to determine the maximum Q-value 
           of all actions based on the observation the environment is in.
            Input:
            obseration
            Output:
            maxq
        """

        self.create_Q_if_new_observation(observation)
        maxq = max(self.Q[observation].values())

        return maxq

    def choose_action(self, observation):
        """This method selects the action to take based on the observation. 
           When the observation is first seen, it initialises the Q values to 0.0.
            Input:
            obseration
            Output:
            action
        """

        self.create_Q_if_new_observation(observation)

        if (random.random() > self.epsilon):
            # explore with 1 - epsilon
            maxQ = self.get_maxQ(observation)

            # multiple actions could have maxQ- pick one at random in that case
            # this is also the case when the Q value for this observation were just set to 0.0
            action = random.choice([k for k in self.Q[observation].keys()
                                    if self.Q[observation][k] == maxQ])

            if (action == 2 and observation[3] == False):
                action = 1
        
        else:
            # explore with epsilon
            action = random.choice(self.valid_actions)

            if (action == 2 and observation[3] == False):
                action = 1

        self.update_parameters()

        return action


    def learn(self, observation, action, reward, next_observation):
        """This method is called after the agent has completed an action and received a reward. 
           This method does not consider future rewards when conducting learning.
            Input:
            obseration
            action
            reward
            next_observation
        """

        self.Q[observation][action] += self.learning_rate * (reward
                                                     + (self.gamma * self.get_maxQ(next_observation))
                                                     - self.Q[observation][action])


#------------------------------------------------------------------------------------------------------------------#
#                                In the following, the SARSA method is implemented.                                #
#------------------------------------------------------------------------------------------------------------------#


def calculate_profit_loss(env, pol, epochs, players):
    """This function calculates the profit or loss.
        Input:
        env: Blackjack Environment
        pol: olicy
        epochs: Number of epochs a player would play
        players: Number of players 
    """

    print("Start with the calculation of the profit or loss ...")

    average_payouts = []

    for player in range(players):   # Simulate different players

        if (player % 100 == 0):  # Display the number of passes at a certain interval
            print("\rPlayer {}/{}".format(player, players), end = "")

        epoch = 1
        total_payout = 0    # store total payout

        while (epoch <= epochs):

            action = np.argmax(pol(env)) 
            obs, payout, complete, _, ddown = env.step(action)

            if (complete == True):
                total_payout += payout
                env.reset()     # New cards for player and dealer
                epoch += 1

        average_payouts.append(total_payout)

    # Show the result in the console and as an image
    avg_total = sum(average_payouts) / players
    print()
    print("Calculation completed!")
    print ("Average payout of a player after {} rounds is {}".format(epochs, avg_total))
    
    plt.rcParams["figure.figsize"] = (18, 9)
    plt.plot(average_payouts, label = "Average Payout for every player")
    plt.axhline(y = avg_total, linestyle = "--", color = "r", label = "Average over all " + str(avg_total))            
    plt.xlabel("Number of players")
    plt.ylabel("Payout after " + str(epochs) + " epochs")
    plt.title("Profit or loss over the complete period")
    plt.grid()
    plt.legend()
    plt.savefig("SARSA_Profit_or_loss_over_the_complete_period.png")
    plt.show()    
    

def create_epsilon_greedy_action_policy(env, q, epsilon):
    """This function create a epsilon greedy action policy.
        Input:
        env: Blackjack Environment
        Q: Q table
        epsilon: Probability of selecting random action instead of the optimal action
        Output:
        policy: function; epsilon greedy action policy with probabilities of each action for each state
    """

    def policy(observation):
        """This function transforms an observation into a action probability.
            Input:
            observation
            Output:
            probability
        """
        # Assign all actions with the same initial probability
        probability = np.ones(env.action_space.n, dtype = float) * epsilon / env.action_space.n  

        best_action = np.argmax(q[observation])  # get best action
        probability[best_action] += (1.0 - epsilon)

        return probability

    return policy


def Agent_SARSA(env, epochs, epsilon, learning_rate, gamma):
    """This function implemented the SARSA Learning Method (on-policy).
        Input:  
        env: Blackjack Environment
        epochs: Number of epochs to be trained
        epsilon: Probability of selecting random action instead of the optimal action
        learning_rate: Learning Rate
        gamma: Gamma discount factor
        Output:
        q: dictionary; mapping of state to action values
        policy: function; transforms an observation into a action probability
    """

    print("Start learning according to the SARSA method ...")

    q = defaultdict(lambda: np.zeros(env.action_space.n))       # Initialise mapping of state to action values
    policy = create_epsilon_greedy_action_policy(env, q, epsilon)  # policy

    for epoch in range(1, epochs + 1):

        if (epoch % 1000 == 0):     # Display the number of passes at a certain interval
            print("\rEpoch {}/{}".format(epoch, epochs), end = "")
            
        current_state = env.reset()
        probs = policy(current_state)   # get epsilon greedy policy
        current_action = np.random.choice(np.arange(len(probs)), p = probs)
        done = False

        while (done == False):
            next_state, reward, done, _, ddown = env.step(current_action)  # calculate next state, reward and done
            next_probs = create_epsilon_greedy_action_policy(env, q, epsilon)(next_state)   # calculate next action probability
            next_action = np.random.choice(np.arange(len(next_probs)), p = next_probs)  # calculate next action

            q_cs_ca = q[current_state][current_action]  # Current state and current action
            td_target = reward + gamma * q[next_state][current_action]
            td_error = td_target - q_cs_ca
            q_cs_ca = q_cs_ca + learning_rate * td_error
            
            if (done == False): # Overwrite only if necessary
                current_state = next_state
                current_action = next_action

    print()
    print("Learning completed!")

    return q, policy

