import gym
from sc2scout.wrapper.action.scout_action import ScoutAction
from s2clientprotocol import sc2api_pb2 as sc_pb
import numpy as np
from PIL import Image


class EvadeActWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(EvadeActWrapper, self).__init__(env)
        self._reverse = False
        self._act = None

    def _reset(self):
        obs = self.env._reset()
        self._reverse = self.judge_reverse()
        self._act = ScoutAction(self.env, self._reverse, move_range=0.5)
        print('evade action,reverse={},move_range=0.2', self._reverse)
        return obs

    def _step(self, action):
        action = self._action(action)
        return self.env._step(action)

    def _action(self, action):
        return self._act.act(action)

    def _reverse_action(self, action):
        raise NotImplementedError()

    def judge_reverse(self):
        home = self.env.unwrapped.owner_base()
        if home[0] < home[1]:
            return False
        else:
            return True


class ZerglingEvadeActLocalWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ZerglingEvadeActLocalWrapper, self).__init__(env)
        self._reverse = False
        self._act = None

    def _reset(self):
        obs = self.env._reset()
        self._reverse = self.judge_reverse()
        self._act = ScoutAction(self.env, self._reverse, move_range=0.5)
        print('evade action,reverse={},move_range=0.2', self._reverse)
        return obs

    def _step(self, action):
        actions = self._action(action)

        # move camera action
        scout = self.env.unwrapped.scout()
        camera_pos = [scout.float_attr.pos_x, scout.float_attr.pos_y]
        print(camera_pos)
        mc_action = self.move_camera(camera_pos)
        actions[0].append(mc_action)

        return self.env._step(actions)

    def _action(self, action):
        return self._act.act(action)

    def _reverse_action(self, action):
        raise NotImplementedError()

    def judge_reverse(self):
        home = self.env.unwrapped.owner_base()
        if home[0] < home[1]:
            return False
        else:
            return True

    def move_camera(self, pos):
        action = sc_pb.Action()
        action.action_raw.camera_move.center_world_space.x = pos[0]
        action.action_raw.camera_move.center_world_space.y = pos[1]
        return action


class ZerglingEvadeActGlobalWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super(ZerglingEvadeActGlobalWrapper, self).__init__(env)
        self._reverse = False
        self._act = None

        self._left_bottom = self.unwrapped._camera_range[0]
        self._right_top = self.unwrapped._camera_range[1]

    def _reset(self):
        obs = self.env._reset()
        self._reverse = self.judge_reverse()
        self._act = ScoutAction(self.env, self._reverse, move_range=0.5)
        print('evade action,reverse={},move_range=0.2', self._reverse)

        self._cur_camera_x = self._left_bottom[0]
        self._cur_camera_y = self._left_bottom[1]

        self._scan_done = False
        self._first_update = True

        map_width = (self._right_top[0] - self._left_bottom[0]) * 3 + 108
        map_height = (self._right_top[1] - self._left_bottom[1]) * 3 + 84
        self._height_map = np.zeros([map_height, map_width], dtype=int)

        return obs

    def _step(self, action):
        actions = self._action(action)

        # move camera action
        mc_action = self.update(self.env.unwrapped._screen_height_map)
        if mc_action is not None:
            actions[0].append(mc_action)

        return self.env._step(actions)

    def _action(self, action):
        return self._act.act(action)

    def _reverse_action(self, action):
        raise NotImplementedError()

    def judge_reverse(self):
        home = self.env.unwrapped.owner_base()
        if home[0] < home[1]:
            return False
        else:
            return True

    def update(self, screen_map):
        if self._scan_done:
            return None

        if self._first_update:
            action = self.move_camera([self._cur_camera_x, self._cur_camera_y])
            self._first_update = False
            return action

        screen_pox_x = (self._cur_camera_x - self._left_bottom[0]) * 3
        screen_pox_y = (self._cur_camera_y - self._left_bottom[1]) * 3

        left_cut = (screen_pox_x + 108 - self._height_map.shape[
            1]) if self._cur_camera_x > self._right_top[0] else 0
        bottom_cut = (screen_pox_y + 84 - self._height_map.shape[
            0]) if self._cur_camera_y > self._right_top[1] else 0
        self.add_map_block(screen_map, screen_pox_x, screen_pox_y, left_cut,
                           bottom_cut)

        if self._cur_camera_y > self._right_top[1] and self._cur_camera_x > \
                self._right_top[0]:
            #self.save_map(self._height_map, 'fullmap.bmp')
            self.convert_feature(32)
            self._scan_done = True

        if self._cur_camera_x >= self._right_top[0]:
            self._cur_camera_x = self._left_bottom[0]
            self._cur_camera_y += 28
        else:
            self._cur_camera_x += 36

        action = self.move_camera([self._cur_camera_x, self._cur_camera_y])
        return action

    def move_camera(self, pos):
        action = sc_pb.Action()
        action.action_raw.camera_move.center_world_space.x = pos[0]
        action.action_raw.camera_move.center_world_space.y = pos[1]
        return action

    def add_map_block(self, map_block, x, y, left_cut, bottom_cut):
        #print('map pos {}:{}, cut {} {}'.format(x, y, left_cut, bottom_cut))

        y = self._height_map.shape[0] - 84 - y
        y = 0 if y < 0 else y

        for i in range(0, map_block.shape[0] - bottom_cut):
            for j in range(map_block.shape[1] - left_cut):
                self._height_map[y+i][x+j] = map_block[i][left_cut + j]

    def save_map(self, map_array, filename):
        for i in range(len(map_array)):
            for j in range(len(map_array[i])):
                print('%3d ' % map_array[i][j], end='')
            print('')

        im = Image.fromarray(np.uint8(map_array))
        # im.show()
        im.save(filename)

    def convert_feature(self, width):
        h, w = self._height_map.shape

        if w > h:
            f = self._height_map[:, int((w-h)/2):int((w+h)/2)]
        else:
            f = self._height_map[int((h-w)/2):int((w+h)/2), :]

        im = Image.fromarray(np.uint8(f))
        im = im.resize((32, 32))

        self.unwrapped._height_feature = np.array(im)
        #self.save_map(height_feature, 'height_feature.bmp')
