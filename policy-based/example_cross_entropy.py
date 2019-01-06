import numpy as np
import gym

env = gym.make('MountainCarContinuous-v0')
env.seed(0)
np.random.seed(0)

print('observation space:', env.observation_space)
print('action space:', env.action_space)
print('  - low:', env.action_space.low)
print('  - high:', env.action_space.high)
