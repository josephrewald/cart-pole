from itertools import count
import torch
import torch.optim as optim
from ray import tune
import torch.nn.functional as F

from agent import Agent
from dqn import DQN
from env_manager import CartPoleEnvManager
from epsilon_greedy_strategy import EpsilonGreedyStrategy
from experience import Experience
from q_values import QValues
from replay_memory import ReplayMemory
from utils import plot, get_moving_average, extract_tensors


def training_function(config):
    batch_size = config['batch_size']
    gamma = config['gamma']
    eps_start = config['eps_start']
    eps_end = config['eps_end']
    eps_decay = config['eps_decay']
    target_update = config['target_update']
    memory_size = config['memory_size']
    lr = config['lr'] 
    num_episodes = config['num_episodes']
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
                #plot(episode_durations, 100)
                break
            
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    plot(episode_durations, 100, config)
    tune.report(average_episode_duration=sum(episode_durations) / len(episode_durations))
    em.close()
