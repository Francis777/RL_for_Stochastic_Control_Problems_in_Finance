from typing import TypeVar, Mapping, Tuple, Callable, Union

# I decide to follow type variable name convention in sample code (/utils/*typevars.py)
# which I think is very explicit and clear

S = TypeVar('S')  # state
A = TypeVar('A')  # action
SSf = Mapping[S, Mapping[S, float]]  # state transition
SSTff = Mapping[S, Mapping[S, Tuple[float, float]]]  # state transition + r(s,s') reward
STSff = Mapping[S, Tuple[Mapping[S, float], float]]  # state transition + R(s) reward
Rf = Mapping[Union[S, Tuple[S, A]], float]  # reward function
Vf = Mapping[S, float]  # value function
QF = Mapping[Tuple[S, A], float]  # value-action function
SAf = Mapping[S, Mapping[A, float]]  # policy
SASf = Mapping[Tuple[S, A, S], Mapping[S, float]]  # state-action transition
SASTff = Mapping[Tuple[S, A], Mapping[S, Tuple[float, float]]]  # state-action transition + r(s,a,s') reward
SATSff = Mapping[Tuple[S, A], Tuple[Mapping[S, float], float]]  # state-action transition + R(s,a) reward
# Note: Another way of specify the MDP is using Mapping[S: Mapping[A: Mapping[S: Tuple[]]]],
# I prefer using Tuple for the state-action pair because it's more straightforward given the definition
# the downside includes: (1) not "incremental" anymore; (2) less support from python dict's build-in functions
