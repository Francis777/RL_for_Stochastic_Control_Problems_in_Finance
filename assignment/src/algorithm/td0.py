import sys
import numpy as np
import gym
from collections import defaultdict


def td0_prediction_v(env, num_episodes, generate_episode, alpha, gamma=1.0):
    V = defaultdict(float)

    for i_episode in range(1, num_episodes + 1):
        # log process
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # generate an episode
        episode = generate_episode(env)
        # unzip the rollouts in this episode
        states, _, rewards = zip(*episode)

        # update value function 
        for i, state in enumerate(states):
            V[state] += alpha * (rewards[i] + gamma * (V[states[i+1]] if i+1 < len(states) else 0) - V[state])

    return V

def td0_prediction_q(env, num_episodes, generate_episode, alpha, gamma=1.0):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # log process
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()

        # generate an episode
        episode = generate_episode(env)
        # unzip the rollouts in this episode
        states, actions, rewards = zip(*episode)

        # update action-value function (every-visit MC)
        for i, state in enumerate(states):
            td_target = rewards[i] + gamma * Q[states[i+1]][actions[i+1]] if i+1 < len(states) else 0
            Q[state][actions[i]] += alpha * (td_target - Q[state][actions[i]])
    return Q