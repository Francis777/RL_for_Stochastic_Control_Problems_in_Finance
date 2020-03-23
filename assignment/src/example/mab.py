# multi-armed bandits
# Note: since there's no suggested assignment for this lecture, I followed Chapter 2 in Sutton-Barto RL book, 
# implemented multi-armed bandit problem with several different action selection algorithm and replicated the figures used in the book


# action-value (sample-average)
# q*(a) = E[ R_t | a_t = a ]
# Q_t(a) = sum of rewards when a taken prior to t / number of times a taken prior to t   ( converge to q*(a) )

# epsilon-greedy action
# A_t = argmax_a Q_t(a) with prob 1 - epsilon, random_a with prob epsilon

from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

class MultiArmedBandit():
    def __init__(self, k: int, epsilon: float, T: int, ucb: bool = False):
        assert(k > 0)
        assert(T > 0)
        self.k = k
        self.T = T
        self.t = 0
        self.epsilon = epsilon
        self.action_value_list: List[float] = [0.] * self.k
        self.true_action_value_list: List[float] = self._generate_q_star()
        self.optimal_action = np.argmax(self.true_action_value_list)
        self.action_counter: List[int] = [0] * self.k
        self.step_size = None
        self.ucb = ucb
        # metrics
        self.reward_list = [0.] * self.T
        self.is_optimal_action = [0] * self.T

    def bandit(self):
        while(self.t < self. T):
            action = self._choose_action()
            reward = self._get_reward(action)

            self._update_q_incremental((action, reward), self.step_size)

            self.reward_list[self.t] = reward
            self.is_optimal_action[self.t] = 1 if action == self.optimal_action else 0

            self.t += 1

    def _choose_action(self):
        if (self.ucb is False):
            if (np.random.rand() > self.epsilon):
                # greedy choice (breaking ties randomly)
                # np.argmax always returns the index of the first max found
                return np.random.choice(np.flatnonzero(self.action_value_list == np.max(self.action_value_list)))
            else:
                # random choice
                return np.random.randint(self.k)
        else:
            # upper-confidence-bound action selection
            c = 2
            if np.all(self.action_counter):
                q_ucb_list = self.action_value_list + c * \
                    np.sqrt(np.divide(np.log(self.t), self.action_counter))
                return np.argmax(q_ucb_list)
            else:
                # return first action with 0 count so far as maximizing action
                return next((idx for idx, count in enumerate(self.action_counter) if count == 0), None)

    def _get_reward(self, action: int):
        mu = self.true_action_value_list[action]
        sigma = 1
        return np.random.normal(mu, sigma)

    def _update_q_incremental(self, last_a_r_pair: Tuple[int, float], step_size: float):
        if (step_size is None):
            # sample average
            self.action_counter[last_a_r_pair[0]] += 1
            self.action_value_list[last_a_r_pair[0]] += (
                last_a_r_pair[1] - self.action_value_list[last_a_r_pair[0]]) / self.action_counter[last_a_r_pair[0]]
        else:
            assert(step_size > 0)
            # constant step size
            self.action_value_list[last_a_r_pair[0]] += step_size * \
                (last_a_r_pair[1] - self.action_value_list[last_a_r_pair[0]])

    def _generate_q_star(self):
        mu = 0
        sigma = 1
        return list(np.random.normal(mu, sigma, self.k))


# suggested by Exercise 2.5
class NonstationaryMultiArmedBandit(MultiArmedBandit):
    def __init__(self, k: int, epsilon: float, T: int, step_size: float = None):
        super().__init__(k, epsilon, T)
        self.step_size = step_size

    def _get_reward(self, action: int):
        # update q*() and optimal action after performing random walk
        mu_rw = 0
        sigma_rw = 0.01
        self.true_action_value_list = list(
            self.true_action_value_list + np.random.normal(mu_rw, sigma_rw, self.k))
        self.optimal_action = np.argmax(self.true_action_value_list)

        mu = self.true_action_value_list[action]
        sigma = 1
        return np.random.normal(mu, sigma)

# plot for mab_fig_1 and mab_fig_2


def plot_1():
    num_steps = 1000
    num_trials = 2000
    epsilon_list = [0, 0.01, 0.1]

    average_reward = np.zeros(
        shape=(len(epsilon_list), num_steps), dtype=float)
    optimal_action = np.zeros(
        shape=(len(epsilon_list), num_steps), dtype=float)
    for i in range(num_trials):
        for idx, epsilon in enumerate(epsilon_list):
            mab = MultiArmedBandit(10, epsilon, num_steps)
            mab.bandit()
            average_reward[idx] += (mab.reward_list -
                                    average_reward[idx]) / (i+1)
            optimal_action[idx] += mab.is_optimal_action
    optimal_action = np.true_divide(optimal_action, num_trials)

    # plot
    # template referenced from https://matplotlib.org/3.1.0/gallery/text_labels_and_annotations/legend_demo.html
    step = np.arange(1, num_steps+1, 1)

    fig1, ax1 = plt.subplots()
    l11, = ax1.plot(step, average_reward[0], 'g')
    l12, = ax1.plot(step, average_reward[1], 'r')
    l13, = ax1.plot(step, average_reward[2], 'b')
    ax1.legend((l11, l12, l13), (r'$\epsilon = 0$ (greedy)', r'$\epsilon = 0.01$',
                                 r'$\epsilon = 0.1$'), loc='lower right', shadow=True)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average reward')

    fig2, ax2 = plt.subplots()
    l21, = ax2.plot(step, optimal_action[0], 'g')
    l22, = ax2.plot(step, optimal_action[1], 'r')
    l23, = ax2.plot(step, optimal_action[2], 'b')
    ax2.legend((l21, l22, l23), (r'$\epsilon = 0$ (greedy)', r'$\epsilon = 0.01$',
                                 r'$\epsilon = 0.1$'), loc='lower right', shadow=True)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Optimal action')
    plt.show()

# plot for non_stat_mab_fig_1 and non_stat_mab_fig_2


def plot_2():
    num_steps = 10000
    num_trials = 2000

    epsilon_list = [0.1]
    alpha_list = [None, 0.1]

    average_reward = np.zeros(
        shape=(len(epsilon_list), len(alpha_list), num_steps), dtype=float)
    optimal_action = np.zeros(
        shape=(len(epsilon_list), len(alpha_list), num_steps), dtype=float)
    for i in range(num_trials):
        for idx1, epsilon in enumerate(epsilon_list):
            for idx2, alpha in enumerate(alpha_list):
                mab = NonstationaryMultiArmedBandit(
                    10, epsilon, num_steps, alpha)
                mab.bandit()
                average_reward[idx1, idx2] += (mab.reward_list -
                                               average_reward[idx1, idx2]) / (i+1)
                optimal_action[idx1, idx2] += mab.is_optimal_action

    optimal_action = np.true_divide(optimal_action, num_trials)

    # plot
    # template referenced from https://matplotlib.org/3.1.0/gallery/text_labels_and_annotations/legend_demo.html
    step = np.arange(1, num_steps+1, 1)

    fig1, ax1 = plt.subplots()
    l11, = ax1.plot(step, average_reward[0, 0], 'g')
    l12, = ax1.plot(step, average_reward[0, 1], 'r')
    ax1.legend((l11, l12), (r'sample average', r'$\alpha = 0.1$'),
               loc='lower right', shadow=True)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average reward')

    fig2, ax2 = plt.subplots()
    l21, = ax2.plot(step, optimal_action[0, 0], 'g')
    l22, = ax2.plot(step, optimal_action[0, 1], 'r')
    ax2.legend((l21, l22), (r'sample average', r'$\alpha = 0.1$'),
               loc='lower right', shadow=True)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Optimal action')
    plt.show()


# plot for mab_fig_3 and mab_fig_4
# comparing UCB action selection vs epsilon-greedy
def plot_3():
    num_steps = 1000
    num_trials = 2000
    default_epsilon = 0.1
    use_ucb = [False, True]

    average_reward = np.zeros(
        shape=(len(use_ucb), num_steps), dtype=float)
    optimal_action = np.zeros(
        shape=(len(use_ucb), num_steps), dtype=float)
    for i in range(num_trials):
        for idx, ucb in enumerate(use_ucb):
            mab = MultiArmedBandit(10, default_epsilon, num_steps, ucb)
            mab.bandit()
            average_reward[idx] += (mab.reward_list -
                                    average_reward[idx]) / (i+1)
            optimal_action[idx] += mab.is_optimal_action
    optimal_action = np.true_divide(optimal_action, num_trials)

    # plot
    # template referenced from https://matplotlib.org/3.1.0/gallery/text_labels_and_annotations/legend_demo.html
    step = np.arange(1, num_steps+1, 1)

    fig1, ax1 = plt.subplots()
    l11, = ax1.plot(step, average_reward[0], 'g')
    l12, = ax1.plot(step, average_reward[1], 'r')
    ax1.legend((l11, l12), (r'$\epsilon-greedy (\epsilon=0.1)$',
                            r'UCB $c = 2$'), loc='lower right', shadow=True)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Average reward')

    fig2, ax2 = plt.subplots()
    l21, = ax2.plot(step, optimal_action[0], 'g')
    l22, = ax2.plot(step, optimal_action[1], 'r')
    ax2.legend((l21, l22), (r'$\epsilon-greedy (\epsilon=0.1)$',
                            r'UCB $c = 2$'), loc='lower right', shadow=True)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Optimal action')
    plt.show()


if __name__ == '__main__':
    plot_3()
