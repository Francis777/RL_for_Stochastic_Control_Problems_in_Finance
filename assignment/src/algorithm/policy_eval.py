from src.process.policy import Policy 
from src.process.mdp import MDP
from src.type_vars import Vf
import numpy as np

# evaluate a policy given full MDP (mathematically V <- V_pi)
def policy_eval(policy: Policy, env: MDP, discount_factor=1.0, theta=1e-6):
    vf: Vf = {s: 0. for s in env.all_states}
    while True:
        delta = 0
        # perform Bellman Expectation Equation for each state
        for s in vf.keys:
            v = 0
            for a, action_prob in enumerate(policy[s].keys):
                for s1 in env.query_successor(s, a):
                    reward = env.query_R(s,a,s1)
                    prob = env.query_Pr(s,a,s1)
                    v += action_prob * prob * (reward + discount_factor * vf[s1])
            delta = max(delta, np.abs(v - vf[s]))
            vf[s] = v
        if delta < theta:
            break
    return vf