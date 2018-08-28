import gym
import numpy as np
from gym.spaces import Box, Tuple

from sc2scout.wrapper.feature.zergling_scout_img_feature_local import ZerglingScoutImgFeatureLocal
from sc2scout.wrapper.feature.zergling_scout_img_feature_global import ZerglingScoutImgFeatureGlobal

class ZerglingScoutEvadeImgObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, is_global=True):
        super(ZerglingScoutEvadeImgObsWrapper, self).__init__(env)
        if is_global:
            self._obs = ZerglingScoutImgFeatureGlobal(32, False)
        else:
            self._obs = ZerglingScoutImgFeatureLocal(32, 12, False)
        self._init_obs_space()

    def _reset(self):
        obs = self.env._reset()
        self._obs.reset(self.env)
        obs = self.observation(obs)
        return obs

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        obs = self.observation(obs)
        return obs, rwd, done, other

    def _init_obs_space(self):
        self.observation_space = self._obs.obs_space()
        print('Evade img obs space=', self._obs.obs_space())

    def _observation(self, obs):
        return self._obs.extract(self.env, obs)


