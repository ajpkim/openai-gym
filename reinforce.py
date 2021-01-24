import argparse
from collections import namedtuple
import pdb
import sys

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from MLP_nets import CategoricalMLP, GaussianMLP
from utils.load_config import load_config


# Original REINFORCE paper: https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf

# REINFORCE is a vanilla policy-gradient based RL algorithm, and
# nearly all policy gradient algorithms are advancements on it.

Transition = namedtuple('Transition', field_names=['obs', 'action', 'reward'])

class REINFORCE_Buffer:
    def __init__(self, gamma):
        self.gamma = gamma
        self.transitions = []
        self.rewards2go = []
        self.trajectory_start = 0
        self.position = 0  # Need to know where trajectory ends for processing discounted rewards2go.

    def push(self, obs, action, reward, done):
        self.transitions.append(Transition(obs, action, reward))
        self.position += 1
        if done:
            self.process_trajectory()

    def discounted_rewards2go(self, rewards):
        rewards2go = [0] * len(rewards)
        for i in range(len(rewards)-1, -1, -1):
            rewards2go[i] = rewards[i] + (rewards2go[i+1] * self.gamma if i + 1 < len(rewards) else 0)
        return rewards2go

    def process_trajectory(self):
        trajectory = self.transitions[self.trajectory_start:self.position]
        rewards = [x.reward for x in trajectory]
        self.rewards2go += self.discounted_rewards2go(rewards)
        self.trajectory_start = self.position

    def get_training_data(self):
        obs, actions, rewards = zip(*self.transitions)
        return {'obs': torch.tensor(obs, dtype=torch.float32),
                'actions': torch.tensor(actions, dtype=torch.float32),
                'rewards': torch.tensor(rewards, dtype=torch.float32),
                'rewards2go': torch.tensor(self.rewards2go, dtype=torch.float32)}

    def clear(self):
        self.transitions.clear()
        self.rewards2go.clear()
        self.position = 0
        self.trajectory_start = 0

    def __repr__(self):
        return "<REINFORCE Buffer>"

def REINFORCE(config):
    """
    REINFORCE (Monte-Carlo policy gradient) implementation.

    args:
      - config: namedtuple that includes the following settings:
        - log directory for TensorBoard logging
        - env
        - env rendering (bool)
        - hidden layer sizes
        - activation function
        - epochs
        - epoch steps
        - gamma (discount rate)
        - learning rate

    returns:
      - dict of losses, returns, and length for all episodes.
    """
    env = gym.make(config.env)
    if isinstance(env.action_space, gym.spaces.Discrete):
        policy_net = CategoricalMLP(env.observation_space.shape[0],
                                    config.hidden_sizes,
                                    env.action_space.n,
                                    config.activation)

    # TODO - write GaussianMLP class
    # elif isinstance(env.action_space, gym.space.Box):
    #     policy_net = GaussianMLP(env.observation_space.shape[0],
    #                              config.hidden_sizes,
    #                              env.action_space.shape[0],
    #                              config.activation)

    optimizer = torch.optim.Adam(params=policy_net.parameters(), lr=config.lr)
    memory_buffer = REINFORCE_Buffer(config.gamma)

    def run_epoch():
        "Gather experience data to train on"
        memory_buffer.clear()
        epoch_steps, episode_rewards = 0, 0
        epoch_rewards = []
        obs = env.reset()

        while True:
            if config.render: env.render()
            action = policy_net(torch.tensor(obs, dtype=torch.float32)).sample().item()
            next_obs, reward, done, _ = env.step(action)
            memory_buffer.push(obs, action, reward, done)
            epoch_steps += 1
            obs = next_obs
            episode_rewards += reward

            if done:
                obs = env.reset()
                epoch_rewards.append(episode_rewards)
                episode_rewards = 0

                if epoch_steps >= config.epoch_steps:
                    if config.render:
                        env.close()
                break
        return np.array(epoch_rewards).mean()

    def compute_loss(obs, actions, weights):
        """
        We want to increase the log probability of highly rewarding actions
        and decrease log probability of unrewarding actions.
        """
        action_log_probs = policy_net.action_log_probs(obs, actions)
        return -(action_log_probs * weights).mean()

    def train():
        optimizer.zero_grad()
        obs, actions, rewards, rewards2go = memory_buffer.get_training_data().values()
        loss = compute_loss(obs, actions, weights=rewards2go)
        loss.backward()
        optimizer.step()
        return

    for epoch in range(config.epochs):
        epoch_mean_reward = run_epoch()
        writer.add_scalar(args.run_name + '_mean_epoch_reward', epoch_mean_reward, epoch)
        train()

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/reinforce.yaml')
    parser.add_argument('--run_name', type=str, default='reinforce_exp')
    args = parser.parse_args()

    config = load_config('config/reinforce.yaml')
    writer = SummaryWriter(config.logdir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Should store configs in some log file for easy matching against TensorBoard runs.
    for key, value in config._asdict().items():
        print(f'{key}: {value}')

    REINFORCE(config)
    writer.close()
