from typing import Generic, Mapping, Sequence 
from src.type_vars import S, A, SAf
import math
import numpy as np

class Policy(Generic[S,A]):
    def __init__(self, data: SAf) -> None:
        try:
            valid: bool = self.check_if_valid(data)
            if not valid:
                raise ValueError()
        except ValueError:
            exit('Input is not a valid policy')

        self.policy_data = data

    def check_if_valid(self, policy: SAf) -> bool:
        for s1 in policy:
            if math.fsum(policy[s1].values()) != 1.0:
                return False
            for a in policy[s1].keys():
                if policy.get(s1).get(a) < 0 or policy.get(s1).get(a) > 1:
                    return False
        return True


    # given a state, return a sampled action according to the distribution
    def sample_action(self, state: S):
        action_candidate: Sequence[A] = []
        action_probability: Sequence[float] = []
        for a, pr in enumerate(self.policy_data[state]):
            action_candidate.append(a)
            action_probability.append(pr)
        a_choice = np.random.choice(action_candidate, 1, p=action_probability)
        return a_choice    

    def get_state_action_probability(self, state: S, action: A) -> float:
        return self.policy_data[state].get(action, 0.)


