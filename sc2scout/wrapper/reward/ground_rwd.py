import math

from sc2scout.wrapper.reward.reward import Reward
from sc2scout.envs import scout_macro as sm

class ZerglingEvadeImpactRwd(Reward):
    def __init__(self, weight=1):
        super(ZerglingEvadeImpactRwd, self).__init__(weight)
        self._scout_last_pos = None

    def reset(self, obs, env):
        scout = env.unwrapped.scout()
        self._scout_last_pos = [scout.float_attr.pos_x, scout.float_attr.pos_y]

    def compute_rwd(self, obs, reward, done, env):
        scout = env.unwrapped.scout()
        scout_current_pos = [scout.float_attr.pos_x, scout.float_attr.pos_y]
        distance = self.distance(self._scout_last_pos, scout_current_pos)
        self._scout_last_pos = scout_current_pos

        if distance < 0.1:
            self.rwd = -2
        else:
            self.rwd = 0

    def distance(self, pos1, pos2):
        x = pos1[0] - pos2[0]
        y = pos1[1] - pos2[1]

        return (x * x + y * y) ** 0.5


class ZerglingEvadeTargetDistanceRwd(Reward):
    def __init__(self, weight=1):
        super(ZerglingEvadeTargetDistanceRwd, self).__init__(weight)
        self._last_target_distance = 0

    def reset(self, obs, env):
        self._last_target_distance = 0

    def compute_rwd(self, obs, reward, done, env):
        scout = env.unwrapped.scout()
        scout_pos = [scout.float_attr.pos_x, scout.float_attr.pos_y]
        enemy_base_pos = env.unwrapped.enemy_base()
        distance = self.distance(scout_pos, enemy_base_pos)

        if distance < self._last_target_distance:
            self.rwd = 1
        else:
            self.rwd = 0

        self._last_target_distance = distance

    def distance(self, pos1, pos2):
        x = pos1[0] - pos2[0]
        y = pos1[1] - pos2[1]

        return (x * x + y * y) ** 0.5

