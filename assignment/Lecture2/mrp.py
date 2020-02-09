from typing import Tuple, Union
from type_vars import S, SSf, SSTff, STSff, Rf, Vf
from mp import MP
import numpy as np


# Note that although the definition of reward function is R(s) = E[R_t+1 | S_t = s]
# there are two types of MRP reward:
#       (1) reward <-> state transition:
#       S_t = s -> S_t+1 = s' leads to different reward R_t+1 for different s'
#       (but in this case R_t+1 still depends ONLY on S_t)
#       (2) reward <-> state:
#       S_t = s -> S_t+1 = s' leads to same reward R_t+1 for any s'
# So in this implementation the two cases are considered as follows:
#       Type (1): input is given as type SSTff, user can call get_expected_reward() for R(s)
#       Type (2): input is given as type STSff, user can call get_state_transition_reward for r(s,s')

class MRP(MP):
    def __init__(self, mrp_input: Union[SSTff, STSff], discount_factor: float) -> None:
        self.type_indicator: bool = MRP.input_type(mrp_input)
        self.gamma: float = discount_factor
        self.state_transition_matrix, self.reward_matrix = self._assign_transition_matrix(
            mrp_input)
        super().__init__(self.state_transition_matrix)
        self.reward_function: Rf = self._assign_reward_function()
        self.value_function: Vf = self._assign_value_function()

    # return True if SSTff, False if STSff
    @staticmethod
    def input_type(mrp_input: Union[SSTff, STSff]) -> bool:
        first_value = mrp_input.get(next(iter(mrp_input)))
        return type(first_value) is dict

    def _assign_transition_matrix(self, mrp_input: Union[SSTff, STSff]) -> Tuple[SSf, SSf]:
        state_transition_matrix = {}
        reward_matrix = {}
        if self.type_indicator is True:
            for s1 in mrp_input:
                state_value = {}
                reward_value = {}
                for s2 in mrp_input.get(s1):
                    state_value.update({s2: mrp_input.get(s1).get(s2)[0]})
                    reward_value.update({s2: mrp_input.get(s1).get(s2)[1]})
                state_transition_matrix.update({s1: state_value})
                reward_matrix.update({s1: reward_value})
        else:
            for s1 in mrp_input:
                state_transition_matrix.update({s1: mrp_input.get(s1)[0]})
                reward_value = {}
                for s2 in mrp_input:
                    reward_value.update({s2: mrp_input.get(s1)[1]})
                reward_matrix.update({s1: reward_value})
        return state_transition_matrix, reward_matrix

    def _assign_reward_function(self) -> Rf:
        reward_list = {}
        for current_state in self.reward_matrix:
            expected_reward = 0
            for next_state in self.state_transition_matrix.get(current_state):
                expected_reward += self.state_transition_matrix.get(current_state).get(next_state, 0.0
                                                                                       ) * self.reward_matrix.get(
                    current_state).get(next_state, 0.0)
            # TODO: find a better way to deal with float point multiplication error
            reward_list.update({current_state: round(expected_reward, 5)})
        return reward_list

    def _assign_value_function(self) -> Vf:
        # V = R + gamma * P * V
        # convert R to np array
        R = np.array([reward for reward in self.reward_function.values()])
        print("R:", R)
        print("state list: ", self.state_list)
        # convert P to np array
        sz = len(self.state_list)
        P = np.empty([sz, sz])
        for index1 in range(sz):
            for index2 in range(sz):
                P[index1, index2] = self.state_transition_matrix.get(self.state_list[index1]).get(self.state_list[
                                                                                                      index2], 0.0)
        # calculate value using matrix inversion
        # https://stackoverflow.com/questions/9155478/how-to-try-except-an-illegal-matrix-operation-due-to-singularity-in-numpy
        print("P: ", P)
        print("A: ", np.eye(sz) - self.gamma * P)
        try:
            V = np.linalg.solve(np.eye(sz) - self.gamma * P, R)
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print("matrix not invertible, will use least square, but most likely result may be incorrect")
                V = np.linalg.lstsq(1 - self.gamma * P, R, rcond=None)[0]
            else:
                raise
        # convert V from np array to dict
        vf = {state: float(value) for state, value in zip(self.state_list, np.nditer(V))}
        return vf

    # R(s)
    def get_expected_reward(self, state: S) -> float:
        return self.reward_function.get(state)

    # r(s, s')
    def get_state_transition_reward(self, current_state: S, next_state: S) -> float:
        return self.reward_matrix.get(current_state).get(next_state, 0.0)

    # v(s)
    def get_value(self, state: S) -> float:
        return self.value_function.get(state)


if __name__ == '__main__':
    # the following example is from the sample code
    input1 = {
        1: ({1: 0.6, 2: 0.3, 3: 0.1}, 7.0),
        2: ({1: 0.1, 2: 0.2, 3: 0.7}, 10.0),
        3: ({3: 1.0}, 0.0)
    }

    input2 = {
        1: {1: (0.6, 7.0), 2: (0.3, 7.0), 3: (0.1, 7.0)},
        2: {1: (0.1, 10.0), 2: (0.2, 10.0), 3: (0.7, 10.0)},
        3: {3: (1.0, 0.0)}
    }

    for mrp_input in (input1, input2):
        print('\n')
        mrp_obj = MRP(mrp_input, 1.0)
        print(mrp_obj.state_transition_matrix)
        print(mrp_obj.reward_matrix)
        print(mrp_obj.reward_function)
        print(mrp_obj.get_expected_reward(1))
        print(mrp_obj.value_function)
        print(mrp_obj.get_value(1))

    # the following example is [Example 6.2 Random Walk] in the RL book
    random_walk_mrp = {
        'T1': {'T1': (1.0, 0)},
        'A': {'T1': (0.5, 0), 'B': (0.5, 0)},
        'B': {'A': (0.5, 0), 'C': (0.5, 0)},
        'C': {'B': (0.5, 0), 'D': (0.5, 0)},
        'D': {'C': (0.5, 0), 'E': (0.5, 0)},
        'E': {'D': (0.5, 0), 'T2': (0.5, 1)},
        'T2': {'T2': (1.0, 0)}
    }
    print('\n')
    mrp_obj = MRP(random_walk_mrp, 1.0)
    print(mrp_obj.state_transition_matrix)
    print(mrp_obj.reward_matrix)
    print(mrp_obj.reward_function)
    print(mrp_obj.value_function)

    # TODO: the solved value function is incorrect (vs the analytical one), figure out why

    # the following example is from David Silver's slide
    student_mrp = {
        'Facebook': ({'Facebook': 0.9, 'Class 1': 0.1}, -1),
        'Class 1': ({'Facebook': 0.5, 'Class 2': 0.5}, -2),
        'Class 2': ({'Sleep': 0.2, 'Class 3': 0.8}, -2),
        'Class 3': ({'Pass': 0.6, 'Pub': 0.4}, -2),
        'Pass': ({'Sleep': 1.0}, 10),
        'Pub': ({'Class 1': 0.2, 'Class 2': 0.4, 'Class 3': 0.4}, 1),
        'Sleep': ({'Sleep': 1.0}, 0)
    }

    print('\n')
    mrp_obj = MRP(student_mrp, 1.0)
    print(mrp_obj.state_transition_matrix)
    print(mrp_obj.reward_matrix)
    print(mrp_obj.reward_function)
    print(mrp_obj.value_function)
