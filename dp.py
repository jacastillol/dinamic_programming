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
    policy = np.zeros([env.nS, env.nA]) / env.nA

    Q = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        Q[s] = q_from_v(env, V, s)
        policy[s][np.argmax(Q[s])] = 1
    
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

def truncated_policy_evaluation(env, policy, V, max_it=1, gamma=1):
    """
    Truncated Policy Evaluation from a policy

    Eval action-value function from value function
        env: modified environment from openAI Gym with access to MDP through env.P
        policy: 2D-array policy[s][a]
        V: 1D-array V[s]
        max_it: stopping criteria
        gamma: discounting factor
    """
    for counter in range(max_it):
        for s in range(env.nS):
            vs = 0
            for a in env.P[s].keys():
                probs, next_states, rewards, dones = zip(*env.P[s][a])
                vs += policy[s][a]*np.sum(probs*(rewards+gamma*V[np.array(next_states)]))
            V[s] = vs

    return V

def truncated_policy_iteration(env, max_it=1, gamma=1, theta=1e-8):
    """
    Truncated Policy Iteration

    Eval action-value function from value function
        env: modified environment from openAI Gym with access to MDP through env.P
        policy: 2D-array policy[s][a]
        max_it: stopping criteria
        gamma: discounting factor
        theta:
    """
    V = np.zeros(env.nS)
    policy = np.zeros([env.nS, env.nA]) / env.nA

    while (True):
        policy =  policy_improvement(env, V, gamma=gamma)
        Vold = V.copy()
        V = policy_evaluation(env, policy, gamma=gamma, theta=theta)
        if max(V-Vold)<theta:
            break

    return policy, V

def value_iteration(env, gamma=1, theta=1e-8):
    V = np.zeros(env.nS)
    
    while True:
        delta = 0
        for s in range(env.nS):
            vs = V[s]
            V[s] = max(q_from_v(env, V, s, gamma))
            delta = max(delta, np.abs(vs-V[s]))
        if delta < theta:
            break
    policy = policy_improvement(env, V, gamma)
    
    return policy, V
