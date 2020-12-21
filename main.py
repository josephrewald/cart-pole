print('Hello from Deep Lizard Cart Pole')
import sys
print(f'Using kernel: {sys.executable}')

import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from agent import Agent
from dqn import DQN
from env_manager import CartPoleEnvManager
from epsilon_greedy_strategy import EpsilonGreedyStrategy
from experience import Experience
from q_values import QValues
from replay_memory import ReplayMemory
from utils import plot, get_moving_average, extract_tensors

from ray import tune

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython: from IPython import display



def training_function(config):
    batch_size = int(256*2)
    gamma = 0.999
    eps_start = 1
    eps_end = 0.01
    eps_decay = 0.001
    target_update = 10
    memory_size = 100000
    lr = config['lr'] #0.001
    num_episodes = 5000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    em = CartPoleEnvManager(device)
    strategy = EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
    agent = Agent(strategy, em.num_actions_available(), device)
    memory = ReplayMemory(memory_size)
    policy_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
    target_net = DQN(em.get_screen_height(), em.get_screen_width()).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
    
    episode_durations = []
    
    for episode in range(num_episodes):
        em.reset()
        state = em.get_state()
        for timestep in count():
            action = agent.select_action(state, policy_net)
            reward = em.take_action(action)
            next_state = em.get_state()
            memory.push(Experience(state, action, next_state, reward))
            state = next_state
                
            if memory.can_provide_sample(batch_size):
                experiences = memory.sample(batch_size)
                states, actions, rewards, next_states = extract_tensors(experiences)
                    
                current_q_values = QValues.get_current(policy_net, states, actions)
                next_q_values = QValues.get_next(target_net, next_states)
                target_q_values = (next_q_values * gamma) + rewards
                    
                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            if em.done:
                episode_durations.append(timestep)
                plot(episode_durations, 100)
                break
            
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
    

    tune.report(mean_loss=sum(episode_durations) / len(episode_durations))
    em.close()

analysis = tune.run(
    training_function, 
    config={
        'lr': tune.grid_search([0.0001, 0.001, 0.01, 0.1])
    },
    mode='max'
    )

# main: 86
# agent: 20
# dqn: 19
# env_manager: 79
# eps greedy: 11
# experience: 5
# moving average: 10
# q values: 20
# replay memory: 20
# utils: 40
# total: approx 300 LOC

# Round 15:
## Episode 3991
## 100 episode moving avg: 72.99
