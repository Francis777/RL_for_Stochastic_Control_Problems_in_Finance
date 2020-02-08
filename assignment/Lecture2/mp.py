# TODO: Check correctness of stantionary distribution

from typing import Generic, Mapping, Sequence
from type_vars import S, SSf
from utils import get_all_states
import numpy as np
from scipy import linalg
# use math.fsum() instead of sum to get rid of rounding error
import math


class MP(Generic[S]):
    def __init__(self, state_transition_matrix: SSf) -> None:
        try:
            valid: bool = self.check_if_valid(state_transition_matrix)
            if not valid:
                raise ValueError()
        except ValueError:
            exit('Input is not a valid Markov Process')

        self.state_list: Sequence[S] = get_all_states(state_transition_matrix)
        self.state_transition_matrix: SSf = state_transition_matrix

    def check_if_valid(self, state_transition_matrix: SSf) -> bool:
        for s1 in state_transition_matrix:
            if math.fsum(state_transition_matrix[s1].values()) != 1.0:
                return False
            for s2 in state_transition_matrix[s1].keys():
                if state_transition_matrix.get(s1).get(s2) < 0 or state_transition_matrix.get(s1).get(s2) > 1:
                    return False
        return True

    def get_stationary_distribution(self) -> Mapping[S, float]:
        sz = len(self.state_list)
        P = np.zeros((sz, sz), dtype=float)
        # construct state transition matrix as a 2-d np array
        for i, s1 in enumerate(self.state_list):
            for j, s2 in enumerate(self.state_list):
                if self.state_transition_matrix.get(s1) is not None:
                    if self.state_transition_matrix[s1].get(s2) is not None:
                        P[i][j] = self.state_transition_matrix[s1][s2]
        # find a row vector v s.t. v * P = v && sum(v) == 1 by solving an overdetermined linear equation
        # ref: https://stephens999.github.io/fiveMinuteStats/stationary_distribution.html
        a = np.concatenate((np.transpose(P) - np.identity(sz), np.ones((1, sz))), axis=0)
        b = np.concatenate((np.zeros((sz, 1)), np.ones((1, 1))), axis=0)
        x = linalg.lstsq(a, b)[0]
        return {s: x[i][0].astype(float) for i, s in enumerate(self.state_list)}


if __name__ == '__main__':
    state_transition_matrix = {
        1: {1: 0.1, 2: 0.6, 3: 0.1, 4: 0.2},
        2: {1: 0.25, 2: 0.22, 3: 0.24, 4: 0.29},
        3: {1: 0.7, 2: 0.3},
        4: {1: 0.3, 2: 0.5, 3: 0.2}
    }
    mp_obj = MP(state_transition_matrix)
    print(mp_obj.state_list)
    print(mp_obj.get_stationary_distribution())
