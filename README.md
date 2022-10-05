# Self-learning Black Jack player based on Reinforcement Learning

This repository contains a self-learning Black Jack player based on reinforcement learning.  
The reinforcement learning methods Q-Learning and SARSA were implemented.  
The Black Jack environment provided by OpenAI is used.  
https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py

The extensions are mostly based on  
https://github.com/ml874/Blackjack--Reinforcement-Learning  

The goal of this repository is to implement the following rules:
1. the Basic Strategy from [1]  
2. the Complete Point-Count System from [1]  
3. in addition to the basic rules, two rule variations    
	- Rewards  
		- Positive Reward times two  
		- Negative Reward times two  
	- Dealer stops  
		- Dealer stops at 16  
		- Dealer stops at 18  

[1] Edward O. Thorp, "Beat the Dealer", Vintage, 1966   


## ToDo:  
Splitting does not work correctly yet. Here there is still room for improvement. 


## Overview of the folder structure and files
| Files               | Description                                                             |
| ------------------- | ----------------------------------------------------------------------- |
| Plots/              | contains all tested hyperparameters and the corresponding images        |
| agent.py            | provides the agents for Q-Learning and SARSA                            |
| deck.py             | provides the deck for card counting                                     |
| environment.py      | provides the environment                                                |
| play.py             | brings everything together here and can be seen as the starting point   |


## Overview of the experiments and achieved results
Drawing & Standing is used in every attempt. The evaluation is based on the average payout. The corresponding image can be found in the Plots folder. A positive result is achieved only once. <br> 
| Learning algorithm   | Doubling Down & Splitting Pairs   | Complete Point-Count System   | Training epochs = Evaluation players   | Evaluation epochs per player   | Average Payout   | Extension   |
| ---------- | ----- | ----- | ----: | ---: | -------: | ------------------- |
| SARSA      | False | False | 100   | 100  | -18.02   |                     |
| SARSA      | False | False | 50000 | 1000 | -186.29  |                     |
| Q-Learning | False | False | 100   | 100  | -30.58   |                     |
| Q-Learning | False | False | 50000 | 1000 | -63.45   |                     |
| Q-Learning | True  | False | 50000 | 1000 | -109.55  |                     |
| Q-Learning | True  | True  | 50000 | 1000 | -103.94  | Number of decks 6   |
| Q-Learning | True  | True  | 50000 | 1000 | -85.10   | Number of decks 4   |
| Q-Learning | True  | True  | 50000 | 1000 | **0.71** | Number of decks 1   |
| Q-Learning | True  | False | 50000 | 1000 | -98.02   | Positive reward * 2 |
| Q-Learning | True  | False | 50000 | 1000 | -269.54  | Negative reward * 2 |
| Q-Learning | True  | True  | 50000 | 1000 | -20.10   | Positive reward * 2 |
| Q-Learning | True  | True  | 50000 | 1000 | -313.13  | Negative reward * 2 |
| Q-Learning | True  | False | 50000 | 1000 | -73.49   | Dealer stops at 16  |
| Q-Learning | True  | False | 50000 | 1000 | -53.40   | Dealer stops at 18  |
| Q-Learning | True  | True  | 50000 | 1000 | -101.18  | Dealer stops at 16  |
| Q-Learning | True  | True  | 50000 | 1000 | -19.52   | Dealer stops at 18  |

