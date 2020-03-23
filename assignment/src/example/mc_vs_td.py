'''
compare MC and TD using the blackjack exmaple

state space: Tuple(Discrete(32), Discrete(11), Discrete(2))
    player's current sum {0, ... , 31}
    dealer's face up {1, ... , 10}, aces can either count as 11 or 1
    whether the player has usebale ace {0, 1}

action space: Discrete(2)
    stick = 0, hit = 1

Detail of this environment: https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
'''

import sys
import gym
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.algorithm.mc import mc_prediction_v, mc_prediction_q, mc_control
from src.algorithm.td0 import td0_prediction_v, td0_prediction_q
from src.algorithm.td_lambda import forward_td_prediction_v, backward_td_prediction_v

from src.utils import plot_policy

def demo_with_random_policy (env):
    for i_episode in range(5):
        print('Game ', i_episode)
        state = env.reset()
        while True:
            print("state: ", state)
            action = env.action_space.sample()
            
            print("action: ", "stick" if action == 0 else "hit")
            state, reward, done, _ = env.step(action)

            if done:
                print("last state: ", state)
                print('End game! Reward: ', reward)
                print('You won :)\n') if reward > 0 else print('You lost :(\n')
                break


# generate episode using naive fixed policy
# default: if sum exceeds 18, "stick" 80% "hit" 20%, vice versa
def generate_episode_naive_policy(env):
    episode = []
    state = env.reset()
    while True:
        probs = [0.8, 0.2] if state[0] > 18 else [0.2, 0.8]
        action = np.random.choice(np.arange(2), p=probs)
        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state
    return episode


def plot_blackjack_values(V):

    def get_Z(x, y, usable_ace):
        if (x,y,usable_ace) in V:
            return V[x,y,usable_ace]
        else:
            return 0

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)
        
        Z = np.array([get_Z(x,y,usable_ace) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(212, projection='3d')
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()


if __name__ == "__main__":
    env = gym.make('Blackjack-v0')

    # demo how the interface works using a random policy 
    demo_with_random_policy(env)

    # MC prediction given a fixed policy 
    V = mc_prediction_v(env, 500000, generate_episode_naive_policy)
    plot_blackjack_values(V)

    # TD(0) prediction given the same fixed policy
    V = td0_prediction_v(env, 500000, generate_episode_naive_policy, alpha=0.1)
    plot_blackjack_values(V)

    # Forward View TD(lambda) prediction given the same fixed policy
    V = forward_td_prediction_v(env, 500000, generate_episode_naive_policy, alpha=0.1, lambd=0.9)
    plot_blackjack_values(V)

    # Backward View TD(lambda) prediction given the same fixed policy
    V = backward_td_prediction_v(env, 500000, generate_episode_naive_policy, alpha=0.1, lambd=0.9)
    plot_blackjack_values(V)

    # GLIE  MC control
    policy, Q = mc_control(env, 600000, alpha=0.01)
    V = dict((k,np.max(v)) for k, v in Q.items())
    plot_blackjack_values(V)
    plot_policy(policy)
