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

ToDo:  
Splitting does not work correctly yet.  

agent.py provides, the agents for Q-Learning and SARSA  
deck.py provides the deck for card counting  
environment.py provides the environment  
play.py brings everything together here and can be seen as the starting point  
Plots This folder contains all tested hyperparameters and the corresponding images.  

[1] Edward O. Thorp, "Beat the Dealer", Vintage, 1966   

