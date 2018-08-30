from gym.spaces import Box

import numpy as np

from sc2scout.wrapper.feature.scout_vec_feature import VecFeature
from sc2scout.wrapper.util.dest_range import DestRange
import sc2scout.envs.scout_macro as sm

MAX_RANGE_SIZE = 360.

class GroundVecFeature(VecFeature):
    def __init__(self):
        super(GroundVecFeature, self).__init__()

    def reset(self, env):
        super(GroundVecFeature, self).reset(env)

    def obs_space(self):
        low = np.zeros(7)
        high = np.ones(7)
        return Box(low, high)

    def extract(self, env, obs):
        scout = env.unwrapped.scout()
        scout_raw_pos = (scout.float_attr.pos_x, scout.float_attr.pos_y)
        home_pos = env.unwrapped.owner_base()
        enemy_pos = env.unwrapped.enemy_base()
        scout_pos = self._pos_transfer(scout_raw_pos[0], scout_raw_pos[1])
        home_pos = self._pos_transfer(home_pos[0], home_pos[1])
        enemy_pos = self._pos_transfer(enemy_pos[0], enemy_pos[1])

        features = []
        features.append(float(scout_pos[0]) / self._map_size[0])
        features.append(float(scout_pos[1]) / self._map_size[1])
        features.append(float(abs(home_pos[0] - scout_pos[0])) / self._map_size[0])
        features.append(float(abs(home_pos[1] - scout_pos[1])) / self._map_size[1])
        features.append(float(abs(enemy_pos[0] - scout_pos[0])) / self._map_size[0])
        features.append(float(abs(enemy_pos[1] - scout_pos[1])) / self._map_size[1])
        features.append(scout.float_attr.facing / MAX_RANGE_SIZE)
        return np.array(features)

class GroundVecFeatureV1(VecFeature):
    def __init__(self):
        super(GroundVecFeatureV1, self).__init__()
        self._dest = None

    def reset(self, env):
        super(GroundVecFeatureV1, self).reset(env)
        self._dest = DestRange(env.unwrapped.enemy_base(), dest_range=25)

    def obs_space(self):
        low = np.zeros(8)
        high = np.ones(8)
        return Box(low, high)

    def extract(self, env, obs):
        scout = env.unwrapped.scout()
        scout_raw_pos = (scout.float_attr.pos_x, scout.float_attr.pos_y)
        home_pos = env.unwrapped.owner_base()
        enemy_pos = env.unwrapped.enemy_base()
        scout_pos = self._pos_transfer(scout_raw_pos[0], scout_raw_pos[1])
        home_pos = self._pos_transfer(home_pos[0], home_pos[1])
        enemy_pos = self._pos_transfer(enemy_pos[0], enemy_pos[1])

        features = []
        features.append(float(scout_pos[0]) / self._map_size[0])
        features.append(float(scout_pos[1]) / self._map_size[1])
        features.append(float(abs(home_pos[0] - scout_pos[0])) / self._map_size[0])
        features.append(float(abs(home_pos[1] - scout_pos[1])) / self._map_size[1])
        features.append(float(abs(enemy_pos[0] - scout_pos[0])) / self._map_size[0])
        features.append(float(abs(enemy_pos[1] - scout_pos[1])) / self._map_size[1])
        features.append(scout.float_attr.facing / MAX_RANGE_SIZE)
        if self._dest.in_range(scout_raw_pos):
            features.append(float(1))
        else:
            features.append(float(0))
        return np.array(features)

