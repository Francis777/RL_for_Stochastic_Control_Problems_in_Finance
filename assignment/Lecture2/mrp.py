from typing import Generic, Mapping, Sequence, Tuple, Union
from type_vars import S, SSf, SSTff, STSff
from mp import MP


# Note that although the definition of reward function is R(s) = E[R_t+1 | S_t = s]
# there are two types of MRP reward:
#       (1) reward <-> state transition:
#       S_t = s -> S_t+1 = s' leads to different reward R_t+1 for different s'
#       (but in this case R_t+1 still depends ONLY on S_t)
#       (2) reward <-> state:
#       S_t = s -> S_t+1 = s' leads to same reward R_t+1 for any s'
# So in this implementation the two cases are considered as follows:
#       Type (1): input is given as type SSTff, user can call get_expected_reward() which converts r(s,s') -> R(s)
#       Type (2): input is given as type STSff, user can call get_reward() which converts R(s) -> r(s,s')

class MRP(MP):
    def __init__(self, input: Union[SSTff, STSff], discount_factor: float) -> None:
        self.type_indicator = MRP.input_type(input)
        self.state_transition_matrix, self.reward_matrix = self._assign_transition_matrix(
            input)
        super().__init__(self.state_transition_matrix)
        self.gamma: float = discount_factor

    # return True if SSTff, False if STSff
    @staticmethod
    def input_type(input: Union[SSTff, STSff]) -> bool:
        first_value = input.get(next(iter(input)))
        return type(first_value) is dict

    def _assign_transition_matrix(self, input: Union[SSTff, STSff]) -> Tuple[SSf, SSf]:
        state_transition_matrix = {}
        reward_matrix = {}
        if self.type_indicator is True:
            for s1 in input:
                state_value = {}
                reward_value = {}
                for s2 in input.get(s1):
                    state_value.update({s2: input.get(s1).get(s2)[0]})
                    reward_value.update({s2: input.get(s1).get(s2)[1]})
                state_transition_matrix.update({s1: state_value})
                reward_matrix.update({s1: reward_value})
        else:
            for s1 in input:
                state_transition_matrix.update({s1: input.get(s1)[0]})
                reward_value = {}
                for s2 in input:
                    reward_value.update({s2: input.get(s1)[1]})
                reward_matrix.update({s1: reward_value})
        return state_transition_matrix, reward_matrix

    def reward_function(state: S) -> float:
        pass

    def get_state_transition_reward(current_state: S, next_state: S) -> float:
        pass

    def value_function(state: S) -> float:
        pass


if __name__ == '__main__':
    # the following example is [Example 6.2 Random Walk] in the RL book
    input1 = {
        1: ({1: 0.6, 2: 0.3, 3: 0.1}, 7.0),
        2: ({1: 0.1, 2: 0.2, 3: 0.7}, 10.0),
        3: ({3: 1.0}, 0.0)
    }
    mrp_obj = MRP(input1, 1.0)
    print("input 1:")
    print(mrp_obj.state_transition_matrix)
    print(mrp_obj.reward_matrix)

    input2 = {
        1: {1: (0.6, 7.0), 2: (0.3, 7.0), 3: (0.1, 7.0)},
        2: {1: (0.1, 10.0), 2: (0.2, 10.0), 3: (0.7, 10.0)},
        3: {3: (1.0, 0.0)}
    }
    mrp_obj = MRP(input2, 1.0)
    print("input 2:")
    print(mrp_obj.state_transition_matrix)
    print(mrp_obj.reward_matrix)
