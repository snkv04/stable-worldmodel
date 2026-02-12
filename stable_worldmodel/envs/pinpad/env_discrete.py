import collections

import gymnasium as gym
import numpy as np

from loguru import logger as logging
import cv2


# TODO: Re-enable targets to be sequences of pads instead of single pads
class PinPadDiscrete(gym.Env):

    COLORS = {
        '1': (255,   0,   0),
        '2': (  0, 255,   0),
        '3': (  0,   0, 255),
        '4': (255, 255,   0),
        '5': (255,   0, 255),
        '6': (  0, 255, 255),
        '7': (128,   0, 128),
        '8': (  0, 128, 128),
    }
    X_BOUND = 16
    Y_BOUND = 16

    def __init__(self, task, target_pad='1', seed=None):
        # Sets up grid
        layout = {
            'three': LAYOUT_THREE,
            'four': LAYOUT_FOUR,
            'five': LAYOUT_FIVE,
            'six': LAYOUT_SIX,
            'seven': LAYOUT_SEVEN,
            'eight': LAYOUT_EIGHT,
        }[task]
        self.layout = np.array([list(line) for line in layout.split('\n')]).T  # Transposes so that actions are (dx, dy)
        assert self.layout.shape == (self.X_BOUND, self.Y_BOUND), (
            f"Layout shape should be ({self.X_BOUND}, {self.Y_BOUND}), got {self.layout.shape}"
        )

        # Sets up pads and spawns
        self.pads = set(self.layout.flatten().tolist()) - set('* #\n')
        self.spawns = []
        for (x, y), char in np.ndenumerate(self.layout):
            if char != '#':
                self.spawns.append((x, y))
        self.target_pad = target_pad

        # To be initialized in reset()
        self.player = None

        # Miscellaneous
        self.random = np.random.RandomState(seed)

    @property
    def action_space(self):
        return gym.spaces.Discrete(5)  # [0, 5)

    @property
    def observation_space(self):
        return gym.spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)

    def reset(self, seed=None, options=None):
        # Selects player location
        if seed is not None:
            self.random = np.random.RandomState(seed)
        self.player = self.spawns[self.random.randint(len(self.spawns))]

        # Gets return values
        obs = self.render()
        info = {'agent_position': np.array(self.player)}
        return obs, info

    def step(self, action):
        # Moves player
        move = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)][action]
        x = np.clip(self.player[0] + move[0], 0, self.X_BOUND - 1)
        y = np.clip(self.player[1] + move[1], 0, self.Y_BOUND - 1)
        tile = self.layout[x][y]
        if tile != '#':
            self.player = (x, y)

        # Gets reward
        reward = 0.0
        if tile == self.target_pad:
            reward += 10.0

        # Makes observation
        obs = self.render()
        terminated = tile == self.target_pad
        truncated = False
        info = {'agent_position': np.array(self.player)}
        return obs, reward, terminated, truncated, info

    def render(self):
        # Sets up grid
        grid = np.zeros((self.X_BOUND, self.Y_BOUND, 3), np.uint8) + 255
        white = np.array([255, 255, 255])
        current = self.layout[self.player[0]][self.player[1]]

        # Colors all cells
        for (x, y), char in np.ndenumerate(self.layout):
            if char == '#':
                grid[x, y] = (192, 192, 192)  # Gray
            elif char in self.pads:
                color = np.array(self.COLORS[char])
                color = color if char == current else (10 * color + 90 * white) / 100
                grid[x, y] = color
        grid[self.player] = (0, 0, 0)

        # Scales up
        image = np.repeat(np.repeat(grid, 14, 0), 14, 1)
        return image.transpose((1, 0, 2))


LAYOUT_THREE = """
################
#1111      3333#
#1111      3333#
#1111      3333#
#1111      3333#
#              #
#              #
#              #
#              #
#              #
#              #
#     2222     #
#     2222     #
#     2222     #
#     2222     #
################
""".strip('\n')

LAYOUT_FOUR = """
################
#1111      4444#
#1111      4444#
#1111      4444#
#1111      4444#
#              #
#              #
#              #
#              #
#              #
#              #
#3333      2222#
#3333      2222#
#3333      2222#
#3333      2222#
################
""".strip('\n')

LAYOUT_FIVE = """
################
#          4444#
#          4444#
#111       4444#
#111           #
#111           #
#111        555#
#           555#
#           555#
#333        555#
#333           #
#333           #
#333       2222#
#          2222#
#          2222#
################
""".strip('\n')

LAYOUT_SIX = """
################
#111        555#
#111        555#
#111        555#
#              #
#              #
#33          66#
#33          66#
#33          66#
#33          66#
#              #
#              #
#444        222#
#444        222#
#444        222#
################
""".strip('\n')

LAYOUT_SEVEN = """
################
#111        444#
#111        444#
#11          44#
#              #
#              #
#33          55#
#33          55#
#33          55#
#33          55#
#              #
#              #
#66          22#
#666  7777  222#
#666  7777  222#
################
""".strip('\n')

LAYOUT_EIGHT = """
################
#111  8888  444#
#111  8888  444#
#11          44#
#              #
#              #
#33          55#
#33          55#
#33          55#
#33          55#
#              #
#              #
#66          22#
#666  7777  222#
#666  7777  222#
################
""".strip('\n')


if __name__ == '__main__':
    import imageio
    from pathlib import Path
    from tqdm import tqdm
    
    env = PinPadDiscrete(task='three')
    logging.info("Made env")

    obs, info = env.reset()
    logging.info(f"Called reset(), and got obs.shape: {obs.shape}")
    frames = [obs]
    
    for i in tqdm(range(3000), desc="Collecting data"):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs)
        
        # if terminated or truncated:
        #     logging.info(f"Episode ended at step {i+1}")
        #     break
    
    # Save video
    video_dir = Path('../stable-worldmodel/videos')
    video_dir.mkdir(exist_ok=True)
    video_path = video_dir / 'pinpad_episode.mp4'
    imageio.mimsave(video_path, frames, fps=30)
    logging.info(f"Saved video with {len(frames)} frames to {video_path}")
