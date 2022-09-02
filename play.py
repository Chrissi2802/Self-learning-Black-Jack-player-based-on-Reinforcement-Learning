#---------------------------------------------------------------------------------------------------#
# File name: play.py                                                                                #
# Autor: Chrissi2802                                                                                #
# Created on: 21.07.2022                                                                            #
#---------------------------------------------------------------------------------------------------#
# Reinforcement Learning - BlackJack Player
# Self-learning BackJack player based on Reinforcement Learning methods
# Exact description in the functions.


import numpy as np
import matplotlib.pyplot as plt
from environment import Blackjack_Env, Blackjack_Env_CC
from agent import Agent_Q, Agent_SARSA, calculate_profit_loss


def train_SARSA(env, epochs, episodes):
    """This function starts the training and evaluation with the SARSA method.
        Input:
        env: Blackjack Environment
        epochs: Number of epochs to be trained
        episodes: Number of players
    """

    Q_SARSA, SARSA_Policy = Agent_SARSA(env, epochs, 0.1, 0.1, 0.95)
    env.reset()
    calculate_profit_loss(env, SARSA_Policy, 1000, episodes)    # Payoff for Off-Policy


def train_Q(env, epochs, players):
    """This function starts the training and evaluation with the Q-Learning method.
        Input:
        env: Blackjack Environment
        epochs: Number of epochs to be trained
        episodes: Number of players
    """

    num_rounds = 1000   # Payout calculated over every episode
    epoch = 1
    total_payout = 0
    average_payouts = []
    dic_actions_count = {}
    dic_actions = {0: "STAND", 1: "HIT", 2: "DOUBLE DOWN", 3: "SPLIT"}
    actions_count = {0:0, 1:0, 2:0, 3:0}   # Number of actions performed in each category

    agent = Agent_Q(env, 1.0, 0.01, 0.1, epochs)
    observation = env.reset()
    
    print("Start learning with Q-Learning and the calculation of the profit or loss ...")
    for sample in range(players):

        if (sample % 100 == 0):  # Display the number of passes at a certain interval
            print("\rPlayer {}/{}".format(sample, players), end = "")

        epoch = 1
        round_payout = 0 # to store total payout over 'num_rounds'
        # Take action based on Q-table of the agent and learn based on that
        while (epoch <= num_rounds):

            action = agent.choose_action(observation)
            #print("Action:", action)            # only for testing
            #print("Observation:", observation)  # only for testing
            next_observation, payout, is_done, _, _ = env.step(action)
            actions_count[action] += 1
            agent.learn(observation, action, payout, next_observation)
            round_payout += payout
            observation = next_observation

            if (is_done):
                observation = env.reset() # Environment deals new cards to player and dealer
                epoch += 1

        total_payout += round_payout
        average_payouts.append(round_payout)

    # Create an understandable dictionary
    for key, value in dic_actions.items():
        dic_actions_count.update({value : actions_count[key]})

    # Show the result in the console and as an image
    avg_total = total_payout / players
    print()
    print("Learning and Calculation completed!")
    print("Average payout of a player after {} rounds is {}".format(num_rounds, avg_total))
    print("Number of actions performed in each category:", dic_actions_count)

    plt.rcParams["figure.figsize"] = (18, 9)
    plt.plot(average_payouts, label = "Average Payout for every player")
    plt.axhline(y = avg_total, linestyle = "--", color = "r", label = "Average over all " + str(avg_total))            
    plt.xlabel("Number of players")
    plt.ylabel("Payout after " + str(num_rounds) + " epochs")
    plt.title("Profit or loss over the complete period")
    plt.grid()
    plt.legend()
    plt.savefig("Q_Learning_Profit_or_loss_over_the_complete_period.png")
    plt.show()   
    

if (__name__ == "__main__"):

    np.random.seed(0)

    epochs = 50000
    episodes = 50000

    # Select one environment
    #env = Blackjack_Env()      # without card counting
    env = Blackjack_Env_CC()    # with card counting

    env.reset(seed = 0)

    # Select one learning algorithm
    train_Q(env, epochs, episodes)      # Q-Learning
    #train_SARSA(env, epochs, episodes) # SARSA

