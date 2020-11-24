from copy import deepcopy
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.trajectories import time_step as ts

class TransCostEnv(py_environment.PyEnvironment):
    """
    The states are d-dim vectors of deviations from the Merton proportion (d is #risky assets)
    state = scale* xi/z
    action = scale*m/z / 10
    The actions are d-dim vectors to indicate the change of state
    """
    def __init__(self, wealth, scale, reward_scale, steps, dRiskyAsset, ValueFn, PropCost, seed=23456):
        self._wealth = wealth
        self._dRiskyAsset = dRiskyAsset
        self._ValFct = ValueFn
        self._dPropCost = PropCost
        self._Nsteps = steps
        self._avg_reward =0.
        self._learningRate_AR = 5e-3
        self._curr_step = 0
        self._reset_next_step = False
        self._scale = scale
        self._reward_scale = reward_scale
        np.random.seed(seed)
        
        self._action_spec = array_spec.BoundedArraySpec(
            shape = (self._dRiskyAsset,), 
            dtype = np.float32,
            minimum = -1.,
            maximum = 1.,
            name = 'action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape = (self._dRiskyAsset,), 
            dtype = np.float32, 
            minimum = -1.,
            maximum = 1.,
            name='observation')
        self._state = np.zeros(shape=(self._dRiskyAsset,), dtype=np.float32)

    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        self._reset_next_step = False
        self._curr_step = 0.
        self._episode_ended = False
        self._state = (2*np.random.random_sample(size=(self._dRiskyAsset,))-1).astype(np.float32)
        return ts.restart(self._state)
    
    def _step(self, action_input):
        if self._reset_next_step:
            return self.reset()

        
        action = action_input.copy()
        
        self._curr_step += 1
        new_state = deepcopy(self._state)
        noise = np.dot(self._ValFct.alphaAtZ(self._wealth), np.random.normal(scale=1./252,size=self._dRiskyAsset))/(self._wealth/self._scale)
        new_state += noise
        new_state += action
        
        if  not array_spec.check_arrays_nest(new_state, self._observation_spec):
            reward_step = -2
        else:
            self._state = new_state
            reward_step = self._reward_fn(action*self._wealth/self._scale)
            
            self._avg_reward += self._learningRate_AR*(reward_step-self._avg_reward)
            reward_step -= self._avg_reward
            reward_step = np.clip(self._reward_scale*reward_step, -1., 1.)
        
        if self._curr_step >= self._Nsteps:
            self._reset_next_step = True
            print('EPISODE AVERAGE REWARD: ', self._avg_reward)
            return ts.termination(observation=self._state, reward=reward_step)

        return ts.transition(observation=self._state, reward=reward_step, discount=1.)
    
    def _reward_fn(self, action):
        reward = - self._ValFct.der2AtX(self._wealth)/2.
        reward *= np.linalg.norm(np.dot(np.transpose(self._ValFct.sigma), self._state*self._wealth/self._scale), 2)**2
        
        if np.linalg.norm(action, 1) != 0.:
            reward += self._ValFct.derAtX(self._wealth) * (1+self._dPropCost*np.linalg.norm(action,1))
        return -(self._scale/self._wealth)*reward