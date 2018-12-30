from collections import deque
import numpy as np
import torch

def dqn_interact(env, agent,
                 n_episodes=2000, window=100, max_t=1000,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """ Deep Q-Learning Agent-Environment interaction.
    
    Params
    ======
        env: instance of OpenAI Gym's environment
        agent: instance of class Agent (see dqn_agent.py for details)
        n_episodes (int): maximum number of training episodes
        window (int): number of episodes to consider when calculating average rewards
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    # all returns
    all_returns = []
    # initialize average rewards
    avg_rewards = deque(maxlen=n_episodes)
    # initialize monitor for most recent rewards
    samp_rewards = deque(maxlen=window)
    # initialize best average reward
    best_avg_reward = -np.inf
    # initialize eps
    eps = eps_start
    # for each episode
    for i_episode in range(1, n_episodes+1):
        # begin the episode
        state = env.reset()
        # initialize the sample reward
        samp_reward = 0
        for t in range(max_t):
            # agent selects an action
            action =  agent.act(state, eps)
            # agent performs the selected action
            next_state, reward, done, _ = env.step(action)
            # agent performs internal updates based on sampled experience
            agent.step(state, action, reward, next_state, done)
            # updated the sample reward
            samp_reward += reward
            # update the state (s-> s') to next time step
            state = next_state
            if done:
                break
        # save final sampled reward
        samp_rewards.append(samp_reward)
        all_returns.append(samp_reward)
        # update epsion with GLIE
        eps = max(eps_end, eps_decay*eps)
        if (i_episode >= 100):
            # get average reward from last and next 100 episodes
            avg_reward = np.mean(samp_rewards)
            # append to deque
            avg_rewards.append(avg_reward)
            # update best average reward
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
        # monitor progress
        message = "\rEpisode {}/{} || Best average reward {} || Epsilon {:.5f} "
        if i_episode % 100 == 0:
            print(message.format(i_episode, n_episodes, best_avg_reward, eps))
        else:
            print(message.format(i_episode, n_episodes, best_avg_reward, eps),end="")
        # stopping criteria
        if np.mean(samp_rewards)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.
                  format(i_episode, np.mean(samp_rewards)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint.pth')
            break

    return all_returns, avg_rewards, best_avg_reward
