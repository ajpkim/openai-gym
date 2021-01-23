from collections import namedtuple

import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from MLP_nets import build_MLP, CategoricalMLP
from utils.load_config import load_config


# https://towardsdatascience.com/policy-gradient-methods-104c783251e0

# Original REINFORCE paper: https://people.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf

# REINFORCE is a vanilla policy-gradient based RL algorithm, and
# nearly all policy gradient algorithms are advancements on it.


## Recap from Lily Weng: https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html#reinforce
# REINFORCE works because the expectation of the sample gradient is
# equal to that of the actual gradient. Thereforce, we can measure rewards
# from real sample trajectories and use them to update the policy gradient.
# We will increase the log probability of actions that are rewarding and
# decrease the log probability of actions that are not.
# It is Monte-Carlo bc it relies on a full trajectory.

# Psuedo Code: REINFORCE
# 1. Initialize policy parameter theta at random
# 2. Play N full episodes, saving (s,a,r,s_prime) transitions
# 3. For every step t of episode k, calculate returns for the subsequent steps
# 4. Calculate loss
# 5. Perform SGD update of weights, minimizing the loss

# A widely used variant of REINFORCE is to subtract a baseline value from
# the return Gt to reduce the variance of the gradient estimate while keeping
# the bias unchanged. A common baseline if to subtract state-value from action-value,
# and if applied we can use advantage A(s,a) = Q(s,a) - V(s) in the gradient update.


Transition = namedtuple('Transition', field_names=['obs', 'action', 'reward'])

class REINFORCE_Buffer:
    def __init__(self, gamma):
        self.gamma = gamma
        self.transitions = []
        self.rewards2go = []
        self.trajectory_start = 0
        self.position = 0  # Used to know where trajectory ends.

    def push(self, obs, action, reward, done):
        self.transitions.append(Transition(obs, action, reward))
        self.position += 1
        if done:
            self.process_trajectory()

    def process_trajectory(self):
        trajectory = self.transitions[self.trajectory_start:self.position]
        rewards = [x.reward for x in trajectory]
        self.rewards2go.append(discounted_rewards2go(rewards, self.gamma))
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

    def __repr__(self):
        return "<REINFORCE Buffer>"

def discounted_rewards2go(rewards, gamma):
    rewards2go = [0] * len(rewards)
    for i in range(len(rewards)-1, -1, -1):
        rewards2go[i] = rewards[i] + (rewards2go[i+1] * gamma if i + 1 < len(rewards) else 0)
    return rewards2go

def compute_loss(action_log_probs, weights):
    return -(action_log_probs * weights).mean()

def REINFORCE(config):
    """
    REINFORCE (Monte-Carlo policy gradient) implementation.

    args:
      - config: namedtuple that includes the following settings:
        - env
        - env render (bool)
        - MLP sizes
        - epochs
        - epoch length
        - gamma (decay rate)
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

    elif isinstance(env.action_space, gym.space.Box):
        policy_net = GaussianMLP(env.observation_space.shape[0],
                                 config.hidden_sizes,
                                 env.action_space.shape[0],
                                 config.activation)

    optimizer = optim.Adam(params=policy_net.parameters(), lr=config.lr)
    memory_buffer = REINFORCE_Buffer(config.gamma)

    def run_epoch():
        "Gather experience data to train on"
        memory_buffer.clear()
        epoch_steps = 0
        obs = env.reset()

        while True:
            if config.render: env.render()
            action = policy_net(torch.tensor(obs, dtype=torch.float32)).sample().item()
            next_obs, reward, done, _ = env.step(action)
            memory_buffer.push(obs, action, reward, done)
            epoch_steps += 1
            obs = next_obs

            if done:
                memory_buffer.process_trajectory()
                obs = env.reset()

                if epoch_steps >= config.epoch_steps:
                    if config.render: env.close()
                    break
        return

    def train():
        optimizer.zero_grad()
        obs, actions, rewards = memory_buffer.get_training_data().values()
        action_log_probs = policy_net.action_log_probs(obs, actions)
        loss = compute_loss(action_log_probs, weights=memory_buffer.rewards2go)
        loss.backward()
        optimizer.step()
        return

    for epoch in range(config.epochs):
        run_epoch()
        train()

    return


if __name__ == '__main__':
    print('yo')
    config = load_config('config/reinforce.yaml')
    print(config)
