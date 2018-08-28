from sc2scout.wrapper.explore_enemy.zerg_scout_wrapper import ZergScoutWrapper
from sc2scout.wrapper.evade_enemy.evade_act_wrapper import EvadeActWrapper
from sc2scout.wrapper.ground_explore.ground_act_wrapper import ZerglingEvadeActLocalWrapper, \
ZerglingEvadeActGlobalWrapper
from sc2scout.wrapper.ground_explore.ground_obs_wrapper import ZerglingScoutEvadeImgObsWrapper, \
GroundImgObsWrapper
from sc2scout.wrapper.ground_explore.ground_rwd_wrapper import ZerglingScoutEvadeImgRwdWrapper
from sc2scout.wrapper.wrapper_factory import WrapperMaker

class ZerglingEvadeMakerLocalV0(WrapperMaker):
    def __init__(self):
        super(ZerglingEvadeMakerLocalV0, self).__init__('ground_v0')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = ZerglingEvadeActLocalWrapper(env)
        env = ZerglingScoutEvadeImgRwdWrapper(env)
        env = ZerglingScoutEvadeImgObsWrapper(env, False)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        return None


class ZerglingEvadeMakerGlobalV0(WrapperMaker):
    def __init__(self):
        super(ZerglingEvadeMakerGlobalV0, self).__init__('ground_v1')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = ZerglingEvadeActGlobalWrapper(env)
        env = ZerglingScoutEvadeImgRwdWrapper(env)
        env = ZerglingScoutEvadeImgObsWrapper(env, True)
        env = ZergScoutWrapper(env)
        return env

    def model_wrapper(self):
        return None


class GroundMakerV2(WrapperMaker):
    def __init__(self):
        super(GroundMakerV2, self).__init__('ground_v3')

    def make_wrapper(self, env):
        if env is None:
            raise Exception('input env is None')
        env = EvadeActWrapper(env)
        print('ground_v2_1')
        env = ZerglingScoutEvadeImgRwdWrapper(env)
        print('ground_v2_2')
        env = GroundImgObsWrapper(env)
        print('ground_v2_3')
        env = ZergScoutWrapper(env)
        print('ground_v2_4')
        return env

    def model_wrapper(self):
        return None


