import time
import gym
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Policy(nn.Module):
    def __init__(self, s_size=4, h_size=16, a_size=2):
        super(Policy, self).__init__()
        # network architecture
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)

    def forward(self, x):
        x =  F.relu(self.fc1(x))
        x =  self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action =  m.sample()
        return action.item(), m.log_prob(action)
        

def reinforce(env, policy, n_episodes=1000, max_t=1000, gamma=1.0, print_every=100):
    """ REINFORCE policy search method

    Params
    ======
         env: Gym Environment class
         policy: Policy class
         n_episodes (int): maximum number of training episodes
         max_t (int): maximum number of timesteps per episode
         gamma (float): discount rate
         print_every (int): how often to print avg. score (over last 1000)
    """
    # initialize monitors
    scores_deque = deque(maxlen=100)
    scores = []
    # search by n completed episodes
    for i_episode  in range(1, n_episodes+1):
        saved_log_probs = []
        rewards = []
        # interact with the environment
        state = env.reset()
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        # store data
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))
        # evaluate return
        discounts = [gamma**i for i in range(len(rewards)+1)]
        R = sum([g*r for g,r in zip(discounts, rewards)])
        # policy improvement
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()
        # backpropagation
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        # monitor output
        print('\rEpisode {}\tAverage Score: {:.2f}'.
              format(i_episode, np.mean(scores_deque)),end='')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.
                  format(i_episode, np.mean(scores_deque)))
        # stop criteria
        if np.mean(scores_deque)>=195.0:
            print('\nEnvironment solved in {:d} iterations!\tAverage Score {:.2f}'.
                  format(i_episode, np.mean(scores_deque)))
            break
    return scores, scores_deque

# create a new environment with a seed
env =  gym.make('CartPole-v0')
env.seed(0)
np.random.seed(0)
# create agent and optimizer
policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
# print state and action spaces dimensions
print('observation space', env.observation_space)
print('action space', env.action_space)
# REINFORCE policy-based method
scores, scores_deque = reinforce(env,policy)
# plot output
scores = []
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
# watch a smart agent
state = env.reset()
for t in range(2000):
    action, _ = policy.act(state)
    env.render()
    time.sleep(0.1)
    state, reward, done, _ = env.step(action)
    if done:
        break 
# close environment
env.close()
