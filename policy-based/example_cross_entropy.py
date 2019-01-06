import numpy as np
from collections import deque
import gym
import torch
import torch.nn as nn

class Agent(nn.Module):
    """ Neural Network Agent

    Params
    ======
    env: its a Gym Environment
    h_size (int): number of neurons of the only hidden layer
    """
    def __init__(self, env, h_size=16):
        super(Agent, self).__init__()
        self.env = env
        # state, hidden layer, action sizes
        self.s_size = env.observation_space.shape[0]         # state size
        self.h_size = h_size                                 # hidden size
        self.a_size = env.action_space.shape[0]              # action size
        # define layers
        self.fc1 = nn.Linear(self.s_size, self.h_size)
        self.fc2 = nn.Linear(self.h_size, self.a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x.cpu().data

    def get_weigths_dim(self):
        return (self.s_size+1)*self.h_size + (self.h_size+1)*self.a_size

    def set_weights(self, weights):
        s_sz = self.s_size
        h_sz = self.h_size)
        a_sz = self.a_size)
        # separate the weights for each layer
        fc1_end = (s_sz*h_sz) + h_sz
        fc1_W = torch.from_numpy(weights[:s_sz *h_sz].reshape(s_sz,h_sz))
        fc1_b = torch.from_numpy(weights[s_sz *h_sz:fc1_end])
        fc2_W = torch.from_numpy(weights[fc1_end:fc1_end+(s_sz *h_sz)].reshape(h_sz,a_sz))
        fc2_b = torch.from_numpy(weights[fc1_end+(h_sz*a_sz):])
        # set the weights for each layer
        self.fc1.weight.data.copy_(fc1_W.view_as(self.fc1.weight.data))
        self.fc1.bias.data.copy_(fc1_b.view_as(self.fc1.bias.data))
        self.fc2.weight.data.copy_(fc2_W.view_as(self.fc2.weight.data))
        self.fc2.bias.data.copy_(fc2_b.view_as(self.fc2.bias.data))

    def evaluate(self, weights, gamma=1.0, max_t=5000):
        self.set_weights(weights)
        episode_return = 0.0
        state = self.env.reset()
        for t in range(max_t):
            state = torch.from_numpy(state).float().to(device)
            action = self.forward(state)
            state, reward, done, _ = self.env.step(action)
            episode_return += reward * math.pow(gamma, t)
            if done:
                break
        return episode_return

def cem(n_iterations=500, max_t=1000, gamma=1.0, print_every=10,
        pop_size=50, elite_frac=0.2, sigma=0.5):
    """ Cross Entropy policy search method

    Params
    ======
         n_iterations (int): maximum number of training iterations
         max_t (int): maximum number of timesteps per episode
         gamma (float): discount rate
         print_every (int): how often to print avg. score (over last 1000)
         pop_size (int): size of population at each iteration
         elite_frac (float): percentage of top performers to use in update
         sigma (float): standard deviation of additive noise
    """
    n_elite = int(pop_size*elite_franc)
    # intialize monitors
    scores_deque = deque(maxlen=100)
    scores = []
    # init weights randomly
    best_weight = sigma*np.random.randn(agent.get_weights_dim())
    for  i_iteration in range(1,n_iterations+1):
        # add a random noise to weights for each population member
        weights_pop = [best_weight + (sigma*np.random.randn(agent.get_weights_dim()))
                       for i in range(pop_size)]
        # evaluate reward for each memeber with a  policy
        rewards = np.array([agent.evaluate(weights, gamma, max_t)
                            for weights in weights_pop])
        # find the best policy from actual population
        elite_idxs = rewards.argsort()[-n_elite:]            # get the best n_elite ind
        elite_weights = [weights_pop[i] for i in elite_idxs] # arrange weights and
        best_weight =  np.array(elite_weights).mean(axis=0)  # for each weight get the best
        # save the weights
        torch.save(agent.state_dict(), 'checkpoint.pth')
        # monitor output
        if i_iteration % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.
                  format(i_iteration, np.mean(scores_deque)))
        # stop criteria
        if np.mean(scores_deque)>=90.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score{:.2f}'.
                  format(i_iteration, np.mean(scores_deque)))
            break

    return scores, scores_deque

# create a new environment with a seed
env = gym.make('MountainCarContinuous-v0')
env.seed(0)
np.random.seed(0)
# print state and action spaces dimensions
print('observation space:', env.observation_space)
print('action space:', env.action_space)
print('  - low:', env.action_space.low)
print('  - high:', env.action_space.high)
# watch how agent performs
state = env.reset()
while True:
    action = [np.random.uniform(low=-1.0,high=1.0)]
    env.render()
    next_state, reward, done, _ = env.step(action)
    state = next_state
    if done:
        break

env.close()
