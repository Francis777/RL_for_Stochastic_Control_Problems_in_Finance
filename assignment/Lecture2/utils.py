# TODO:
#   1. add memorization decorator

from typing import Mapping, Sequence, Any
from type_vars import S


def get_all_states(state_transition_matrix: Mapping[S, Any]) -> Sequence[S]:
    state_list = [state for state in state_transition_matrix.keys()]
    return state_list
