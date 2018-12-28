import numpy as np
from collections import defaultdict

class Agent:

    def __init__(self, nA=6):
        """ Initialize agent.
        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.Q = defaultdict(lambda: np.zeros(self.nA))
        self.eps = 1.0
        self.gamma = 0.99
        self.alpha = 0.01

    def glie_eps(self, i):
        eps_min = 0.001
        self.eps = max(self.eps*0.999,eps_min)
        # self.eps = max(1.0/i,0.005)
        # thresh = 1
        # self.eps = (thresh-min(i,thresh))/thresh*(1-eps_min)+eps_min

    def select_action(self, state):
        """ Given the state, select an action.
        Params
        ======
        - state: the current state of the environment
        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        return np.argmax(self.Q[state]) if np.random.random() > self.eps else np.random.choice(self.nA)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.
        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        Qsa_next = np.max(self.Q[next_state]) if next_state is not None else 0
        target = reward + self.gamma * Qsa_next
        self.Q[state][action] = self.Q[state][action] + self.alpha * (target - self.Q[state][action])
