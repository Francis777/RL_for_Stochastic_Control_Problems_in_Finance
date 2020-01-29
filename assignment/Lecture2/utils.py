# TODO:
#   1. add memorization decorator

from typing import Mapping, Set, Any
from type_vars import S


def get_all_states(state_transition_matrix: Mapping[S, Any]) -> Set[S]:
    return set(state_transition_matrix.keys())
