import unittest
import numpy as np
from frozenlake import FrozenLakeEnv

def policy_evaluation_soln(env, policy, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            Vs = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_state, reward, done in env.P[s][a]:
                    Vs += action_prob * prob * (reward + gamma * V[next_state])
            delta = max(delta, np.abs(V[s]-Vs))
            V[s] = Vs
        if delta < theta:
            break
    return V

env = FrozenLakeEnv()
random_policy = np.ones([env.nS, env.nA]) / env.nA

class Tests(unittest.TestCase):

    def policy_evaluation(self, policy_evaluation):
        soln = policy_evaluation_soln(env, random_policy)
        to_check = policy_evaluation(env, random_policy)
        np.testing.assert_array_almost_equal(soln, to_check)
