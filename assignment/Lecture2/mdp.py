# TODO:
#   (1) how to implement MDP "incrementally" based on MRP?
#   (2) how to "plug-in" policy?

from typing import Generic, Union, Sequence, Tuple
from type_vars import S, A, SASf, SASTff, SATSff


class MDP(Generic[S, A]):
    def __init__(self, mdp_input: Union[SASTff, SATSff], discount_factor: float) -> None:
        self.type_indicator: bool = MDP.input_type(mdp_input)
        self.gamma: float = discount_factor
        self.states, self.actions = self._get_all_states_and_actions(mdp_input)
        self.transition_matrix: Tuple[SASf, SASf] = self._assign_transition_matrix(mdp_input)

    # return True if SASTff, False if SATSff
    # TODO: unify with the same method in MRP
    @staticmethod
    def input_type(mdp_input: Union[SASTff, SATSff]) -> bool:
        first_value = mdp_input.get(next(iter(mdp_input)))
        return type(first_value) is dict

    def _get_all_states_and_actions(self, mdp_input: Union[SASTff, SATSff]) -> Tuple[Sequence[S], Sequence[A]]:
        state_list = []
        action_list = []
        for sa_tuple in mdp_input:
            if sa_tuple[0] not in state_list:
                state_list.append(sa_tuple[0])
            if sa_tuple[1] not in action_list:
                action_list.append(sa_tuple[1])

        return state_list, action_list

    def _assign_transition_matrix(self, mdp_input: Union[SASTff, SATSff]):
        print(1)
        return {}, {}

    def get_mrp(self):
        pass


if __name__ == '__main__':
    # the following example is from the sample code
    mdp_input = {
        (1, 'a'): ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
        (1, 'b'): ({2: 0.3, 3: 0.7}, 2.8),
        (1, 'c'): ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2),
        (2, 'a'): ({1: 0.3, 2: 0.6, 3: 0.1}, 5.0),
        (2, 'c'): ({1: 0.2, 2: 0.4, 3: 0.4}, -7.2),
        (3, 'a'): ({3: 1.0}, 0.0),
        (3, 'b'): ({3: 1.0}, 0.0)
    }
    mdp_obj = MDP(mdp_input, 1.0)
    print(mdp_obj.states)
    print(mdp_obj.actions)
