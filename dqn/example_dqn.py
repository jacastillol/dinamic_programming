from dqn_agent import Agent
from dqn_monitor import dqn_interact
import gym

env = gym.make('LunarLander-v2')
agent = Agent(state_size=8, action_size=4, seed=0)
all_returns, avg_reward, best_avg_reward = dqn_interact(env, agent)
