import numpy as np

def policy_evaluation(env, policy, gamma=1, theta=1e-8):
    """
    Policy Evaluation algorithm
    
    Eval state-value function using Bellman Expectation equation 
        env: modified environment from openAI Gym with access to MDP through env.P
        policy: 2D-narray probability representation P[s][a]
        gamma: discounting factor
        theta: stopping criteria
    """
    V = np.zeros(env.nS)

    delta = np.inf
    while delta > theta:
        delta = 0
        for s in range(env.nS):
            vs = 0
            for a in env.P[s].keys():
                probs, next_states, rewards, dones = zip(*env.P[s][a])
                vs += policy[s][a]*np.sum(probs*(rewards+gamma*V[np.array(next_states)]))
                delta = max(delta, np.abs(vs-V[s]))
                V[s] = vs

    return V
