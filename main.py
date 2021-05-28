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
from training_function import training_function

# from ray import tune


batch_size =    [128] #, 256, 512] # 256 - number of replay memory experiences considered
gamma =         [0.9] #, 0.99, 0.999] # 0.999 - discount factor in Bellman eqn
eps_start =     [1] # 1 - exploration rate
eps_end =       [0.01] # 0.01
eps_decay =     [0.01] # , 0.001, 0.0001] # 0.001
target_update = [10] # 10 - frequency for updating target network
memory_size =   [100000] # 100000 - capacity of replay memory
lr =            [0.0001] # 0.001 - learning rate
num_episodes =  [1000] # 1000

# analysis = tune.run(
#     training_function, 
#     config={
#         'batch_size':       tune.grid_search(batch_size),
#         'gamma':            tune.grid_search(gamma),
#         'eps_start':        tune.grid_search(eps_start),
#         'eps_end':          tune.grid_search(eps_end),
#         'eps_decay':        tune.grid_search(eps_decay),
#         'target_update':    tune.grid_search(target_update),
#         'memory_size':      tune.grid_search(memory_size),
#         'lr':               tune.grid_search(lr),
#         'num_episodes':     tune.grid_search(num_episodes),
#     },
#     mode='max'
#     )


config={
    'batch_size':       batch_size[0],
    'gamma':            gamma[0],
    'eps_start':        eps_start[0],
    'eps_end':          eps_end[0],
    'eps_decay':        eps_decay[0],
    'target_update':    target_update[0],
    'memory_size':      memory_size[0],
    'lr':               lr[0],
    'num_episodes':     num_episodes[0],
    }

training_function(config)
