import numpy as np
from model import Actor
import torch
import torch.optim as optim

GAMMA = 0.99       # discount factor
LR = 5e-4          # learning rate

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
        self.optimizer = optim.Adam(self.actor_local.parameters(), lr=LR)

    def step(self, state, action, reward, next_state, done):
        """ Interact Agent and Environment.
        """
        experiences = (state, action, reward, next_state, done)
        self.learn(experiences, GAMMA)

    def act(self, state, eps=0):
        """ Returns actions for a given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float), for epsilon-greedy action selection
        """
        # NN evaluation to get stochastic policy (turn off training mode)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action_values = self.actor_local(state)
        self.actor_local.train()
        # Epsilon-greedy action selection
        if np.random.rand() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.nA))

    def learn(self, experiences, gamma):
        """ Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_target_next = self.actor_local(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.actor_local(states).gather(1,actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

