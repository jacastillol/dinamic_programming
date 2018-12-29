import numpy as np
from model import Actor
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """ This Agent interacts and learn from the environment."""
    def __init__(self, state_size, action_size, seed):
        """ Initialize agent object. 
        
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        self.nA = action_size
        self.nS = state_size

        # Actor
        self.actor_local = Actor(self.nS, self.nA, seed).to(device)

    def step(self, state, action, reward, next_state, done):
        """ Interact
        """

    def act(self, state, eps=0):
        """ Returns actions for a given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float), for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action_values = self.actor_local(state)
        self.actor_local.train()
        if np.random.rand() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.nA))

    def learn(self, experience, gamma):
        """ Update value parameters using given batch of experience tuples
        """


