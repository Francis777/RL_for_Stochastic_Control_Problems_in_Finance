# The design of this interface(naming convention) is heavily referenced to openai gym
# https://github.com/openai/gym/blob/c0860592f51b53c00ce57e00cb3a9195943847d9/gym/core.py

from typing import Sequence
from src.type_vars import S, A
from src.process.policy import Policy

# Note that Env is a base class where a specific environment should inherit from
# In later examples I directly use the existing environments in openai gym(e.g. blackjack, cartpole etc.) for convenience
class Env(object):

    reward_range = (-float('inf'), float('inf'))
    action_space: Sequence[A] = None
    state_space: Sequence[S] = None

    def step(self, state, action):
        """
        As Prof. Rao suggested, this method is the essence of the interface, which is a mapping from (state, action) to a sample of (next state, reward)
        Args:
            state (S): current state of the agent
            action (A): an action provided by the agent
        Returns:
            next_state (object): the next state of the agent
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended(reach terminating state or reach goal)
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        Returns:
            state (object): the initial state for the next episode.
        """
        raise NotImplementedError


    def generate_epsiode(self, policy: Policy):
        episode = []
        state = self.reset()
        while True:
            action = policy(state)
            next_state, reward, done = self.step(state, action)
            episode.append((state, action, reward))
            if done: 
                break
            state = next_state
        return episode



