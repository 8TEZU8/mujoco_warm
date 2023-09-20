import gym
import warm
import numpy as np
import sys

from gym.envs.registration import load_env_plugins as _load_env_plugins
from gym.envs.registration import make, register, registry, spec

from tkinter.tix import InputOnly
import torch as tor
from torch import nn
from pathlib import Path
from collections import deque
import matplotlib.pyplot as plt
import random, time, datetime, os, copy

from SAC_class import SAC
from SAC_class import Logger

# Hook to load plugins from entry points
_load_env_plugins()


# Classic
# ----------------------------------------

register(
    id="Warm-v0",
    entry_point="warm:WarmEnv",
    reward_threshold=1000.0,
)

use_cuda = tor.cuda.is_available()
print(f"Using CUDA: {use_cuda}")
print()

save_dir_f = Path("checkpoints_front") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir_f.mkdir(parents=True)
save_dir_b = Path("checkpoints_back") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
save_dir_b.mkdir(parents=True)

#パラメータ定義
action_dim = 2
episode = 5
state_idx = [0,1,2,3,4,5,6,7,8,9]#学習に使う状態量
record_every = 100
render_end = 5

#環境の構築
env = gym.make("Warm-v0",render_mode="rgb_array")

sac_f = SAC(len(state_idx), action_dim, save_dir_f)
sac_b = SAC(len(state_idx), action_dim, save_dir_b)

logger_f = Logger(save_dir_f)
logger_b = Logger(save_dir_b)

initpos = np.zeros(27)
direction = 1
first_task = "front"
task = first_task

front_counter = 0
back_counter = 0

action_gain = 20
action_max = 4095
action_min = 0
before_actions = np.array([(action_max+action_min)/2 , (action_max+action_min)/2])

max_episode_steps=1000
steps = 0

for e in range(episode):
    
    steps = 0
    state = env.reset()
    state = env.reset_model(initqpos=initpos)
    state = state[state_idx]
    
    if task == "front":
        front_counter = front_counter+1
        while True:
            action = [float(i) for i in sac_f.act(state)]
            
            action = np.clip(np.array(action)*action_gain+before_actions, action_min, action_max)
            action = ((2*action/(action_max-action_min))-1)

            next_state, reward, done, direction, info = env.step(action)
            next_state = np.array(next_state)
            next_state = next_state[state_idx].tolist()
            
            reward = reward[0]+reward[1]
            
            if e > episode-render_end:
                logger_f.log_video(env.render())

            sac_f.cache(state, next_state, action, reward, done)

            q, p_loss, q_loss = sac_f.learn()

            logger_f.log_step(reward, p_loss, q_loss, q)

            state = next_state
            
            steps += 1
            if steps > max_episode_steps:
                done = True

            if done:
                break
                
    elif task == "back":
        back_counter = back_counter+1
        while True:
            action = [float(i) for i in sac_b.act(state)]
            
            action = np.clip(np.array(action)*action_gain+before_actions, action_min, action_max)
            action = ((2*action/(action_max-action_min))-1)

            next_state, reward, done, direction, info = env.step(action)
            next_state = np.array(next_state)
            next_state = next_state[state_idx]
            
            reward = -1*reward[0]+reward[1]
        
            if e > episode-render_end:
                logger_b.log_video(env.render())

            sac_b.cache(state, next_state, action, reward, done)

            q, p_loss, q_loss = sac_b.learn()

            logger_b.log_step(reward, p_loss, q_loss, q)

            state = next_state
            
            steps += 1
            if steps > max_episode_steps:
                done = True

            if done:
                break
                
    else:
        sys.exit("Error: don't have task")
        
    
    initpos = env.get_initpos()
    
    #multitask learner
    if direction == -1:
        task = "back"
    else:
        task = "front"
    
    logger_f.log_episode()
    logger_b.log_episode()

    if e % record_every == 0:
        logger_f.record(episode=e, alpha=float(sac_f.log_alpha.logalpha), step=int(sac_f.curr_step))
        logger_b.record(episode=e, alpha=float(sac_b.log_alpha.logalpha), step=int(sac_b.curr_step))
    
    print(f"\r episode:"+("O"*int(20*e/episode))+("-"*(20-int(20*e/episode)))+f"{e}/{episode}:{task}", end="")

logger_f.release_video()
logger_b.release_video()
print(f"\n front rearning:{front_counter}")
print(f"back rearning:{back_counter}")
print("finished")
env.close()#環境を終了 