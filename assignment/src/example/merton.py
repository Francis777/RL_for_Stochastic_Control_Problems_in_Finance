# TODO: reformat MertonPortfolio as Env(openai gym format) so that we can directly plug in all other RL algorithms
from typing import NamedTuple, Callable, Mapping, Tuple
import numpy as np


# The structure of class MertonPortfolio references:
# https://github.com/coverdrive/MDP-DP-RL/blob/master/src/examples/port_opt/merton_portfolio.py and
# https://github.com/coverdrive/MDP-DP-RL/blob/master/src/examples/port_opt/port_opt.py
class MertonPortfolio():
    def __init__(self, expiry, rho, r, mu, sigma, epsilon, gamma, ts):
        self.expiry = expiry  # T
        self.time_steps = ts
        self.delta_t = self.expiry / self.time_steps
        self.rho = rho  # utility discount rate
        self.r = r  # rate for riskless asset
        self.mu = mu  # mean of risky asset rate
        self.sigma = sigma  # variance of risky asset rate
        self.epsilon = epsilon  # bequest function param
        self.gamma = gamma  # utility function param
        self.num_risky = 1
        self.riskless_returns = [self.r * self.delta_t] * self.time_steps
        self.returns_gen_funcs: Callable[[int], np.array] = [
            lambda n, delta_t=self.delta_t: self.risky_returns_gen(n, delta_t)] * self.time_steps
        self.cons_util_func: Callable[[float],
                                      float] = lambda x: self.cons_utility(x)
        self.beq_util_func: Callable[[float],
                                     float] = lambda x: self.beq_utility(x)
        self.discount_rate = self.rho * self.delta_t
        self.init_state = (0, 1.)  # t, W_t

    '''
    Below are helper functions that return analytical result and set assumptions
    referenced from:
    http://web.stanford.edu/class/cme241/lecture_slides/MertonPortfolio.pdf
    https://github.com/Francis777/CME241-Assignment/tree/master/assignment/writeup/merton.pdf
    '''

    def get_optimal_allocation(self) -> float:
        # note that here we fix the number of risky assets to 1
        return (self.mu - self.r) / (self.sigma ** 2 * self.gamma)

    def get_nu(self) -> float:
        num = (self.mu - self.r)*(self.get_optimal_allocation())
        return self.rho / self.gamma - (1. - self.gamma) *\
            (num / (2. * self.gamma) + self.r / self.gamma)

    def get_optimal_consumption(self, t) -> float:
        nu = self.get_nu()
        if nu == 0:
            opt_cons = 1. / (self.expiry - t + self.epsilon)
        else:
            opt_cons = nu / (1. + (nu * self.epsilon - 1) *
                                np.exp(-nu * (self.expiry - t)))
        return opt_cons

    def risky_returns_gen(self, samples: int, delta_t: float) -> np.ndarray:
        return np.random.normal(
            self.mu - 0.5 * self.sigma**2 * delta_t,
            self.sigma * np.sqrt(delta_t),
            samples
        )

    def cons_utility(self, x: float) -> float:
        return x ** (1. - self.gamma) / (1. - self.gamma) if self.gamma != 1. else np.log(x)

    def beq_utility(self, x: float) -> float:
        return self.epsilon ** self.gamma * self.cons_utility(x)

    """
    Below are helper functions for sampling from the MDP 
    """
    # sample actions from current policy
    def sample_actions_gen(self, params, num_samples):
        # sample consumption
        mu, nu = params[:, 2]
        cons_samples = np.random.beta(mu*nu, (1 - mu) * nu, num_samples)
        # sample allocation
        mean = params[-2]
        sigma = params[-1]
        alloc_samples = np.random.normal(mean, sigma, num_samples)
        return [tuple(x) for x in np.vstack(cons_samples + alloc_samples).T]

    # sample reward and next state given current state and action
    def state_reward_gen(self, state, action, num_samples):
        t, W = state
        cons = action[0]
        risky_alloc = action[1]
        riskless_alloc = 1. - risky_alloc
        alloc = np.array([riskless_alloc, risky_alloc])
        ret_samples = np.hstack((
            np.full((num_samples, 1), self.riskless_returns[t]),
            self.returns_gen_funcs[t](num_samples)
        ))
        next_W = [W * (1. - cons) * max(1e-8,
                                        alloc.dot(np.exp(rs)))
                  for rs in ret_samples]
        return [((t + 1, w), self.cons_util_func(W * cons) + (np.exp(-self.discount_rate) * self.beq_util_func(w) if t == self.time_steps - 1 else 0.)) for w in next_W]


    def test_fixed_policy_with_optimal_action(self, num_episode):
        returns = []
        for _ in range(num_episode):
            state = self.init_state
            ret = 0.
            for i in range(self.time_steps):
                # select (analytical optimal) action 
                cons = self.get_optimal_consumption(state[0]/self.time_steps * self.expiry)
                alloc = self.get_optimal_allocation()
                state, reward = self.state_reward_gen(state, (cons,alloc), 1)[0]
                ret += reward * np.exp(-self.discount_rate * i)
            returns.append(ret)
        return sum(returns) / len(returns)


if __name__ == '__main__':

    test = MertonPortfolio(
        expiry=0.4,
        rho=0.04,
        r=0.04,
        mu=0.08,
        sigma=0.03,
        epsilon=1e-8,
        gamma=0.2,
        ts=5
    )

    opt_alloc = test.get_optimal_allocation()
    print(opt_alloc)

    ret = test.test_fixed_policy_with_optimal_action(5000)
    print(ret)