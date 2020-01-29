from typing import TypeVar, Mapping

# Type variable names follow the convention in sample code
S = TypeVar('S')  # state
SSf = Mapping[S, Mapping[S, float]]  # state transition probability matrix
