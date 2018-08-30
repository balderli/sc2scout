import gym
from sc2scout.wrapper.reward import ground_rwd as gr

class GroundExploreRwd(gym.Wrapper):
    def __init__(self, env):
        super(GroundExploreRwd, self).__init__(env)
        self._rewards = None

    def _assemble_reward(self):
        raise NotImplementedError

    def _reset(self):
        self._assemble_reward()
        obs = self.env._reset()
        for r in self._rewards:
            r.reset(obs, self.env.unwrapped)
        return obs

    def _step(self, action):
        obs, rwd, done, other = self.env._step(action)
        new_rwd = 0
        for r in self._rewards:
            r.compute_rwd(obs, rwd, done, self.env.unwrapped)
            new_rwd += r.rwd
        return obs, new_rwd, done, other

class ZerglingScoutEvadeImgRwdWrapper(GroundExploreRwd):
    def __init__(self, env):
        super(ZerglingScoutEvadeImgRwdWrapper, self).__init__(env)

    def _assemble_reward(self):
        self._rewards = [
            gr.ZerglingEvadeImpactRwd(),
            gr.ZerglingEvadeTargetDistanceRwd()
        ]

class GroundRwdWrapper(GroundExploreRwd):
    def __init__(self, env):
        super(GroundRwdWrapper, self).__init__(env)

    def _assemble_reward(self):
        self._rewards = [
            gr.GroundImpactRwd(),
            gr.GroundDistanceRwd()
        ]

class GroundRwdWrapperV1(GroundExploreRwd):
    def __init__(self, env):
        super(GroundRwdWrapperV1, self).__init__(env)

    def _assemble_reward(self):
        self._rewards = [
            gr.GroundFinalRwd(),
            gr.GroundExploreTargetRwd()
        ]

class GroundRwdWrapperV2(GroundExploreRwd):
    def __init__(self, env):
        super(GroundRwdWrapperV2, self).__init__(env)

    def _assemble_reward(self):
        self._rewards = [
            gr.GroundFinalRwd(),
            gr.GroundRangeRwd()
        ]


