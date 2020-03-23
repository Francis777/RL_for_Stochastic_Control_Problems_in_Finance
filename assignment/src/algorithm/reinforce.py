import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical


# here we use a neural network as the policy function approximator
# both to capture the nonlinearity as well as the convenience of taking gradient
class Policy(nn.Module):
    def __init__(self, nS, nA):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(nS, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, nA)

    # since the action space is dicrete, we use softmax policy
    def forward_pass(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward_pass(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def reinforce(n_episodes, T, gamma=1.0):

    for i_episode in range(1, n_episodes+1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        for t in range(T):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break

        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([a*b for a, b in zip(discounts, rewards)])

        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()


if __name__ == '__main__':
    # test REINFORCE implementation using mountain car env
    torch.manual_seed(0)
    env = gym.make('MountainCar-v0')
    env.seed(0)
    device = torch.device("cpu")
    policy = Policy(2, env.action_space.n).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-2)
    reinforce(n_episodes=1000, T=1000)

    # test the learned policy
    state = env.reset()
    for t in range(1000):
        action, _ = policy.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        if done:
            break

    env.close()
