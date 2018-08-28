import numpy as np
from gym.spaces import Box
from PIL import Image

class GroundImgFeature:
    def __init__(self, file_name):
        self._im_data = np.array(Image.open(file_name))

    def reset(self, env):
        pass

    def extract(self, env, obs):
        return self._im_data

    def obs_space(self):
        low = np.zeros(self._im_data.shape)
        high = np.ones(self._im_data.shape)
        return Box(low, high)


if __name__ == '__main__':
    im = Image.open("height_feature.bmp")
    print('im=', im)
    arr = np.array(im)
    for i in range(0, 32):
        print('arr=', arr[i])
    print('shape=', arr.shape)

