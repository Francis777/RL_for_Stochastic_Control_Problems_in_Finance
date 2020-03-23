from src.process.policy import Policy, random_policy
from src.process.mdp import MDP
from src.algorithm.policy_eval import policy_eval
from src.type_vars import S, A, Vf
import numpy as np

# iteratively perform policy evaluation(estimate v_pi) and policy improvement(generate v_pi' >= v_pi)
# note that this is the very basic version of PI, in GPI both components can be changed
def policy_iteration(env: MDP, discount_factor=1.0):
    # initialize random policy
    policy = random_policy(env)
    
    while True:
        # policy evaluation
        vf: Vf = policy_eval(policy, env, discount_factor)
        policy_stable: bool = True

        for s in range(env.all_states):
            # best action under current policy
            chosen_a = np.argmax(policy[s])
            
            # find the best action by one-step lookahead
            action_values = [0.] * range(env.all_actions)
            for a, action_prob in enumerate(policy[s].keys):
                for s1 in env.query_successor(s, a):
                    reward = env.query_R(s,a,s1)
                    prob = env.query_Pr(s,a,s1)
                    action_values[a] += action_prob * prob * (reward + discount_factor * vf[s1])
            best_a = np.argmax(action_values)
                          
            # greedy update, check if policy stable(unchanged for all state)
            if chosen_a != best_a:
                policy_stable = False
            policy[s] = np.eye(env.nA)[best_a]
        
        if policy_stable:
            return policy, vf

