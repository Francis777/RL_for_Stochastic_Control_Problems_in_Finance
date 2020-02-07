from typing import TypeVar, Mapping, Tuple, Callable

# I decide to follow type variable name convention in sample code (/utils/*typevars.py)
# which I think is very explicit and clear

S = TypeVar('S')  # state
SSf = Mapping[S, Mapping[S, float]]  # state transition
SSTff = Mapping[S, Mapping[S, Tuple[float, float]]]  # state transition + r(s,s') reward
STSff = Mapping[S, Tuple[Mapping[S, float], float]]  # state transition + R(s) reward
Rf = Mapping[S, float] # reward function
Vf = Mapping[S, float] # value function