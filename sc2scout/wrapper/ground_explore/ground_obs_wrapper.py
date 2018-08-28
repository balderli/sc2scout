import gym
import numpy as np
from gym.spaces import Box, Tuple

from sc2scout.wrapper.feature.zergling_scout_img_feature_local import ZerglingScoutImgFeatureLocal
from sc2scout.wrapper.feature.zergling_scout_img_feature_global import ZerglingScoutImgFeatureGlobal
from sc2scout.wrapper.feature.ground_img_feature import GroundImgFeature
from sc2scout.wrapper.feature.ground_vec_feature import GroundVecFeature

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


class GroundImgObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(GroundImgObsWrapper, self).__init__(env)
        self._global = GroundImgFeature("height_feature.bmp")
        self._vec = GroundVecFeature()
        self._init_obs_space()
        print("TargetObsWrapperV3: g_shape={};v_shape={};total_obs_shape={}".format(
              self._global.obs_space().shape, self._vec.obs_space().shape,
              self.observation_space.shape))

    def _reset(self):
        obs = self.env._reset()
        self._global.reset(self.env)
        self._vec.reset(self.env)
        obs = self.observation(obs)
        return obs

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        obs = self.observation(obs)
        return obs, rwd, done, other

    def observation(self, obs):
        g_img = self._global.extract(self.env, obs)
        vec = self._vec.extract(self.env, obs)
        return np.hstack([g_img.flatten(), vec])

    def _get_dim(self, ob_space):
        shape_size = len(ob_space.shape)
        dim = 1
        for i in range(0, shape_size):
            dim = dim * ob_space.shape[i]
        return dim

    def _init_obs_space(self):
        g_dim = self._get_dim(self._global.obs_space())
        v_dim = self._get_dim(self._vec.obs_space())
        low =  np.zeros(g_dim + v_dim)
        high = np.ones(g_dim + v_dim)
        self.observation_space = Box(low, high)
        print("obs space", self.observation_space)


