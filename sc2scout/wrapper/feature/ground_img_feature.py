import numpy as np
from gym.spaces import Box
from PIL import Image

class GroundImgFeature:
    def __init__(self, resolution=32, channel=7):
        self._resolution = resolution
        self._channel = channel

    def reset(self, env):
        pass

    def extract(self, env, obs):
        mini_map = obs.observation['feature_minimap']
        #print('minimap_shape=', mini_map.shape)
        return self._trans_img(mini_map)

    def obs_space(self):
        low = np.zeros([self._resolution, self._resolution, self._channel])
        high = np.ones([self._resolution, self._resolution, self._channel])
        return Box(low, high)

    def _trans_img(self, mini_map):
        img = np.zeros([self._resolution, self._resolution, self._channel])
        img[:,:,0] = mini_map.height_map
        img[:,:,1] = mini_map.visibility_map
        img[:,:,2] = mini_map.creep
        img[:,:,3] = mini_map.camera
        img[:,:,4] = mini_map.player_id
        img[:,:,5] = mini_map.player_relative
        img[:,:,6] = mini_map.selected
        return img

