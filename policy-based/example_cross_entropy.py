import numpy as np
from collections import deque
import gym

class Agent:
    def get_weigths_dim(self):
        pass
    def evaluate(self, weights, gamma=1.0, max_t=5000):
        pass

def cem(n_iterations=500, max_t=1000, gamma=1.0, print_every=10,
        pop_size=50, elite_frac=0.2, sigma=0.5):
    ''' Cross Entropy policy search method

    Params
    ======
         n_iterations (int): maximum number of training iterations
         max_t (int): maximum number of timesteps per episode
         gamma (float): discount rate
         print_every (int): how often to print avg. score (over last 1000)
         pop_size (int): size of population at each iteration
         elite_frac (float): percentage of top performers to use in update
         sigma (float): standard deviation of additive noise
    '''
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
