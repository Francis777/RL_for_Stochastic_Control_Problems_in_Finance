from src.type_vars import Vf, Pf, S, A
from src.process.mdp import MDP
from typing import TypeVar, Mapping, Tuple, Generic, Sequence, Union, NoReturn

def valueIteration(self, mdp: MDP, epsilon: float = 1e-10) -> Tuple[Vf, Pf, Mapping[S, Sequence[float]]]:
    # initialize value function and policy
    vf: Vf = {s: 0. for s in mdp.all_states}
    pi: Pf = {}
    q_value = {}
    while True:
        # compute v_k+1 given v_k
        vf_new: Vf = {}
        for s in mdp.all_states:
            if s in mdp.terminal_states:
                vf_new.update({s: mdp.query_R(s, None)})
            else:
                # compute Q(s,a) for all possible a given s
                q_max = None
                for a in mdp.all_actions:
                    q = mdp.query_R(s, a) + mdp.gamma * sum([mdp.query_Pr(s, a, s1) * vf[s1] for s1 in
                                                                mdp.query_successor(s, a)])
                    q_max = q if q_max is None else max(q, q_max)
                # update V(s)
                vf_new.update({s: q_max})
        # check for convergence
        if max([abs(vf[state] - vf_new[state]) for state in mdp.all_states]) < epsilon:
            break
        vf = vf_new

    # retrieve optimal policy after value function converges
    for state in mdp.non_terminal_states:
        # calculate optimal q value for all possible a given s
        opt_q = [mdp.reward.get((state, action)) + mdp.gamma * sum([mdp.state_transition_matrix.get((
            state, action)).get(s1, 0.) * vf.get(s1) for s1 in mdp.all_states]) for action in
                    mdp.all_actions]
        pi.update({state: mdp.all_actions[opt_q.index(max(opt_q))]})
        q_value.update({state: opt_q})
    return vf, pi, q_value