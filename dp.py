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


def q_from_v(env, V, s, gamma=1):
    """
    Q-value from V-value
    
    Eval action-value function from value function 
        env: modified environment from openAI Gym with access to MDP through env.P
        s: 1D-narray of the V fuction
        s: current state
        gamma: discounting factor
    """
    q = np.zeros(env.nA)
        
    for i,a in enumerate(env.P[s].keys()):
        probs, next_states, rewards, dones = zip(*env.P[s][a])
        q[i] = np.sum(probs*(rewards+gamma*V[np.array(next_states)]))

    return q

def policy_improvement(env, V, gamma=1):
    """
    Policy improvement from V-value
    
    Eval action-value function from value function 
        env: modified environment from openAI Gym with access to MDP through env.P
        V: 1D-narray of the V fuction
        gamma: discounting factor
    """

    Q = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        Q[s] = q_from_v(env, V, s)
        policy[s][argmax(Q[s])] = 1
    
    return policy

def policy_iteration(env, gamma=1, theta=1e-8):
    """
    Policy Iteration algorithm for getting an optimal policy
    
    Eval action-value function from value function 
        env: modified environment from openAI Gym with access to MDP through env.P
        gamma: discounting factor
        theta: stopping criteria
    """
    policy = np.ones([env.nS, env.nA]) / env.nA
    
    V = np.zeros(env.nS)
    is_policy_stable = False
    while (True):
        V = policy_evaluation(env, policy, gamma=gamma, theta=theta)
        policy_ =  policy_improvement(env, V, gamma=gamma)
        if np.all( policy==policy_ ):
            is_policy_stable = True
        policy = policy_
        if is_policy_stable:
            break

    return policy, V
