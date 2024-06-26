import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import json
import os
import random
import sys
from itertools import product
from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from dm_env import specs
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

import dmc
import utils
from logger import Logger
from replay_buffer import make_replay_loader
from video import VideoRecorder

torch.backends.cudnn.benchmark = True


def get_domain(task):
    if task.startswith('point_mass_maze'):
        return 'point_mass_maze'
    return task.split('_', 1)[0]


def get_data_seed(seed, num_data_seeds):
    return (seed - 1) % num_data_seeds + 1


def eval(global_step, agent, env, logger, num_eval_episodes, video_recorder):
    step, episode, total_reward = 0, 0, 0
    eval_until_episode = utils.Until(num_eval_episodes)
    while eval_until_episode(episode):
        time_step = env.reset()
        video_recorder.init(env, enabled=(episode == 0))
        while not time_step.last():
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(time_step.observation, step=global_step, eval_mode=True)
            time_step = env.step(action)
            video_recorder.record(env)
            total_reward += time_step.reward
            step += 1

        episode += 1
        video_recorder.save(f'{global_step}.mp4')

    episode_reward = total_reward / episode
    with logger.log_and_dump_ctx(global_step, ty='eval') as log:
        log('episode_reward', episode_reward)
        log('episode_length', step / episode)
        log('step', global_step)

    return episode_reward


def set_seed(cfg):
    if cfg.seed is None:
        cfg.seed = random.randint(0, 100000)
    utils.set_seed_everywhere(cfg.seed)


@hydra.main(config_path='config', config_name='config_cds')
def main(cfg: DictConfig):
    work_dir = Path.cwd()
    print(f'workspace: {work_dir}')
    ori_work_dir = Path(get_original_cwd())
    sys.path.append(get_original_cwd())

    set_seed(cfg)

    # create logger
    logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.wandb)

    # create envs
    env = dmc.make(cfg.task, seed=cfg.seed)

    # create agent
    agent = hydra.utils.instantiate(cfg.agent, obs_shape=env.observation_spec().shape,
        action_shape=env.action_spec().shape, num_expl_steps=0)

    replay_dir_list_main = []
    replay_dir_list_share = []

    share_tasks = []
    for task, data_type in zip(cfg.share_task, cfg.data_type):
        datasets_dir = ori_work_dir / cfg.replay_buffer_dir      # 存储数据的目录
        replay_dir = datasets_dir.resolve() / Path(task+"-td3-"+str(data_type)) / 'data'
        print(f'replay dir: {replay_dir}')
        if task == cfg.task:
            replay_dir_list_main.append(replay_dir)
        else:
            replay_dir_list_share.append(replay_dir)
            share_tasks.append(task)

    print("CDS.  load main dataset..", cfg.task)
    replay_loader_main = make_replay_loader(env, replay_dir_list_main, cfg.replay_buffer_size,
                cfg.batch_size // 2, cfg.replay_buffer_num_workers, cfg.discount,      # batch size (half)
                main_task=cfg.task, task_list=[cfg.task])
    replay_iter_main = iter(replay_loader_main)      # run OfflineReplayBuffer.sample function

    print("CDS.  load share dataset..", share_tasks)
    replay_loader_share = make_replay_loader(env, replay_dir_list_share, cfg.replay_buffer_size,
                cfg.batch_size // 2 * 10, cfg.replay_buffer_num_workers, cfg.discount,  # batch size是10倍，后取top10
                main_task=cfg.task, task_list=share_tasks)
    replay_iter_share = iter(replay_loader_share)     # run OfflineReplayBuffer.sample function
    print("load data done.")

    # create video recorders
    video_recorder = VideoRecorder(work_dir if cfg.save_video else None)

    timer = utils.Timer()
    global_step = 0
    best_reward = 0

    train_until_step = utils.Until(cfg.num_grad_steps)
    eval_every_step = utils.Every(cfg.eval_every_steps)
    log_every_step = utils.Every(cfg.log_every_steps)

    if cfg.wandb:
        path_str = f'{cfg.agent.name}_{cfg.share_task[0]}_{cfg.share_task[1]}_{cfg.data_type[0]}_{cfg.data_type[1]}'
        wandb_dir = f"./wandb/{path_str}_{cfg.seed}"
        if not os.path.exists(wandb_dir):
            os.makedirs(wandb_dir)
        wandb.init(project="RL_project", entity="virlus_yu")
        for key, value in cfg.items():
            wandb.config.update({key: value})
        wandb.run.name = path_str

    while train_until_step(global_step):
        # try to evaluate
        if eval_every_step(global_step):
            logger.log('eval_total_time', timer.total_time(), global_step)
            reward = eval(global_step, agent, env, logger, cfg.num_eval_episodes, video_recorder)
            if reward > best_reward:
                best_reward = reward
                agent.save(work_dir)

        # train the agent
        metrics = agent.update(replay_iter_main, replay_iter_share, global_step, cfg.num_grad_steps)

        # log
        logger.log_metrics(metrics, global_step, ty='train')
        if log_every_step(global_step):
            elapsed_time, total_time = timer.reset()
            with logger.log_and_dump_ctx(global_step, ty='train') as log:
                log('fps', cfg.log_every_steps / elapsed_time)
                log('total_time', total_time)
                log('step', global_step)

        global_step += 1


if __name__ == '__main__':
    main()
