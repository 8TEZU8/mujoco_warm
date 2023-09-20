import numpy as np
import sys

from gym import utils
from gym.envs.mujoco import MuJocoPyEnv
from gym.spaces import Box

from typing import Optional, Union

class WarmEnv(MuJocoPyEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(self, **kwargs):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)
        MuJocoPyEnv.__init__(
            self, 
            "warm.xml", #model path 
            5, #skip frame
            observation_space=observation_space, #observation space
            **kwargs
        )
        utils.EzPickle.__init__(self, **kwargs)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        a = [i*27.79 for i in a] #add ctrlgain (change at xmlfile)
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
    
        reward_fwd = (xposafter - xposbefore) / self.dt
            
        reward_ctrl = -ctrl_cost_coeff * np.square(a).sum()
        reward = [reward_fwd, reward_ctrl]
        done = False
        if self.data.sensordata[0] or self.data.sensordata[1]:
            done = True
        
        if xposafter > 0:
            direction = -1
        else:
            direction = 1
        
        ob = self._get_obs()

        if self.render_mode == "human":
            self.render()

        return (
            ob,
            reward,
            done,
            direction,
            dict(reward_fwd=reward_fwd, reward_ctrl=reward_ctrl),
        )

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        
        return np.concatenate([qpos.flat[2:5], qpos.flat[13:15], qvel.flat[:5], qvel.flat[13:15]])
    
    def get_initpos(self):        
        return self.sim.data.qpos.flat[:]

    def reset_model(self, initqpos = None):
        if initqpos is not None:
            self.set_state(
                self.init_qpos+initqpos,
                #+ self.np_random.uniform(low=-1.0, high=1.0, size=self.model.nq),
                self.init_qvel,
                #+ self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv),
            )
        else:
            self.set_state(
                self.init_qpos,
                #+ self.np_random.uniform(low=-1.0, high=1.0, size=self.model.nq),
                self.init_qvel,
                #+ self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv),
            )
        return self._get_obs()