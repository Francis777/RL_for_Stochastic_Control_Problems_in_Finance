import sys
import numpy as np
from collections import defaultdict


def forward_td_prediction_v(env, num_episodes, generate_episode, alpha, lambd, gamma=1.0):

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

        # helper for calculating lambda-return 
        lambda_returns_factor = np.array([(1 - lambd) * lambd**i for i in range(len(rewards)+1)])
        # update value function 
        for i, state in enumerate(states):
            returns = [sum(rewards[i:i+k+1]) for k in range(len(states) - i)]
            V[state] += alpha * (sum(returns * lambda_returns_factor[:len(returns)]) - V[state])

    return V

# TODO: this implementation is very inefficient due to iterating over dict V and E in each iteration
# find a way to have both structural uniformity and efficiency 
def backward_td_prediction_v(env, num_episodes, generate_episode, alpha, lambd, gamma=1.0):
    def update_eligibility_trace(E, curr_s):
        for s in E:
            E[s] *= gamma * lambd
            if s == curr_s:
                E[s] += 1
        return E

    def update_V(V, alpha, delta, E):
        for s in V:
            V[s] += alpha * delta * E[s]
        return V

    V = defaultdict(float)
    E = defaultdict(float) # eligibility trace
    
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
        for i , state in enumerate(states):
            E = update_eligibility_trace(E, state)
            delta = rewards[i] + gamma * (V[states[i+1]] if i+1 < len(states) else 0) - V[state]
            V = update_V(V, alpha, delta, E)

    return V