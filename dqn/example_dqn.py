from dqn_agent import Agent
from dqn_monitor import dqn_interact
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch

env = gym.make('LunarLander-v2')
agent = Agent(state_size=8, action_size=4, seed=0)
all_returns, avg_reward, best_avg_reward = dqn_interact(env, agent)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(all_returns)), all_returns)
plt.ylabel('Scores')
plt.xlabel('Episode #')
plt.show()

# load the weights from file
agent.actor_local.load_state_dict(torch.load('checkpoint.pth'))

# check performance of the agent
for i in range(10):
    state = env.reset()
    score = 0
    for j in range(1000):
        action = agent.act(state)
        env.render()
        state, reward, done, _ = env.step(action)
        score += reward
        if done:
            break
    print('Episode: {} || Score: {}'.format(i+1,score))
