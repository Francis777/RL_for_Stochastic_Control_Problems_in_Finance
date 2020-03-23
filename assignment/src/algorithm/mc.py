import sys
import numpy as np
import gym
from collections import defaultdict


def mc_prediction_v(env, num_episodes, generate_episode, gamma=1.0):
    N = defaultdict(float)
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

        # helper for calculating discount
        discounts = np.array([gamma**i for i in range(len(rewards)+1)])

        # update value function (every-visit MC)
        for i, state in enumerate(states):
            N[state] += 1.0
            V[state] += (sum(rewards[i:] * discounts[:-(1+i)]) - V[state]) / N[state]

    return V


def mc_prediction_q(env, num_episodes, generate_episode, gamma=1.0):
    N = defaultdict(lambda: np.zeros(env.action_space.n))
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

        # helper for calculating discount
        discounts = np.array([gamma**i for i in range(len(rewards)+1)])

        # update action-value function (every-visit MC)
        for i, state in enumerate(states):
            N[state][actions[i]] += 1.0
            Q[state][actions[i]] += (sum(rewards[i:]*discounts[:-(1+i)]) - Q[state][actions[i]]) / N[state][actions[i]]
    return Q


def mc_control(env, num_episodes, alpha, gamma=1.0):
    # generate episode following epsilon-greedy policy
    def generate_episode_epsilon_greedy(env, Q, epsilon, nA):
        def get_probs(Q_s, epsilon, nA):
            policy_s = np.ones(nA) * epsilon / nA
            best_a = np.argmax(Q_s)
            policy_s[best_a] = 1 - epsilon + (epsilon / nA)
            return policy_s

        episode = []
        state = env.reset()
        while True:
            action = np.random.choice(np.arange(nA), p=get_probs(Q[state], epsilon, nA)) \
                                        if state in Q else env.action_space.sample()
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            state = next_state
            if done:
                break
        return episode

    # update Q estimate using the most recent episode (constant step size)
    def update_Q(env, episode, Q, alpha, gamma):
        states, actions, rewards = zip(*episode)
        discounts = np.array([gamma**i for i in range(len(rewards)+1)])
        for i, state in enumerate(states):
            Q[state][actions[i]] += alpha*(sum(rewards[i:]*discounts[:-(1+i)]) - Q[state][actions[i]])
        return Q

    nA = env.action_space.n
    Q = defaultdict(lambda: np.zeros(nA))
    for i_episode in range(1, num_episodes+1):
        # log progress
        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes), end="")
            sys.stdout.flush()
        
        # adjust epsilon to 1/k according to GLIE
        epsilon = 1 / i_episode
        episode = generate_episode_epsilon_greedy(env, Q, epsilon, nA)
        Q = update_Q(env, episode, Q, alpha, gamma)

    # final deterministic policy corresponding to the final action-value function estimate
    opt_policy = dict((k,np.argmax(v)) for k, v in Q.items())
    return opt_policy, Q