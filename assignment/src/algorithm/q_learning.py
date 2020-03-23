import sys
import gym
import numpy as np
import random
import math
from collections import defaultdict


# note that this function is the same with SARSA except we use  R + gamma * max(Q(s', ~)) as TD target
def q_learning(env, num_episodes, alpha, gamma=1.0):
    def epsilon_greedy(Q, state, nA, eps):
        if random.random() > eps:
            return np.argmax(Q[state])
        else:
            return random.choice(np.arange(nA))

    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))

    for i_episode in range(1, num_episodes+1):

        state = env.reset()

        # adjust epsilon according to GLIE
        eps = 1.0 / i_episode

        action = epsilon_greedy(Q, state, nA, eps)

        while True:
            next_state, reward, done, _ = env.step(action)
            if not done:
                # update Q(curr_state, curr_action)
                target = reward + \
                    (gamma * (np.max(Q[next_state])
                              if next_state is not None else 0))
                Q[state][action] += alpha * (target - Q[state][action])

                state = next_state
                action = next_action
            if done:
                Q[state][action] += alpha * (reward - Q[state][action])
                break

    return Q
