import glob
import numpy as np
from omegaconf import DictConfig
import hydra
from hydra.utils import get_original_cwd
import dmc
from pathlib import Path

class Agent:
    # An example of the agent to be implemented.
    # Your agent must extend from this class (you may add any other functions if needed).
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
    def act(self, state):
        action = np.random.uniform(-5, 5, size=(self.action_dim))
        return action
    def load(self, load_path):
        pass


def load_data(data_path):
    """
    An example function to load the episodes in the 'data_path'.
    """
    epss = sorted(glob.glob(f'{data_path}/*.npz'))
    episodes = []
    for eps in epss:
        with open(eps, 'rb') as f:
            episode = np.load(f)
            episode = {k: episode[k] for k in episode.keys()}
            episodes.append(episode)
    print(len(episodes))
    return episodes

load_data("collected_data/walker_run-td3-medium/data")
load_data("collected_data/walker_run-td3-medium-replay/data")
load_data("collected_data/walker_walk-td3-medium/data")
load_data("collected_data/walker_walk-td3-medium-replay/data")

def eval(eval_env, agent, eval_episodes):
    """
    An example function to conduct online evaluation for some agentin eval_env.
    """
    returns = []
    for episode in range(eval_episodes):
        time_step = eval_env.reset()
        cumulative_reward = 0
        while not time_step.last():
            action = agent.act(time_step.observation, 0, eval_mode=True)
            time_step = eval_env.step(action)
            cumulative_reward += time_step.reward
        returns.append(cumulative_reward)
    return sum(returns) / eval_episodes

@hydra.main(config_path='config', config_name='config_cds')
def main(cfg: DictConfig):
    task_name = "walker_walk"
    seed = 42
    eval_env = dmc.make(task_name, seed=seed)
    
    walk_agent = hydra.utils.instantiate(cfg.agent, obs_shape=eval_env.observation_spec().shape,
        action_shape=eval_env.action_spec().shape, num_expl_steps=0)
    
    walk_path = Path(get_original_cwd())
    walk_path = walk_path / 'checkpoints/06-02-02-50-walker_walk-Share_walker_walk_walker_run-medium-dw_cds'
    
    walk_agent.load(load_path=walk_path)
    
    print(eval(eval_env=eval_env, agent=walk_agent, eval_episodes=10))

    task_name = "walker_run"
    seed = 42
    eval_env = dmc.make(task_name, seed=seed)
    
    run_agent = hydra.utils.instantiate(cfg.agent, obs_shape=eval_env.observation_spec().shape,
        action_shape=eval_env.action_spec().shape, num_expl_steps=0)
    
    run_path = Path(get_original_cwd())
    run_path = run_path / 'checkpoints/06-02-15-05-walker_run-Share_walker_run_walker_walk-medium-dw_cds'
    
    run_agent.load(run_path)
    print(eval(eval_env=eval_env, agent=run_agent, eval_episodes=10))
    
    

if __name__ == '__main__':
    main()