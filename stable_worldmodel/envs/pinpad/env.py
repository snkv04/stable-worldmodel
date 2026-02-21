import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw

from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.pinpad import assets as pinpad_assets
from stable_worldmodel.envs.pinpad.constants import (
    COLORS,
    X_BOUND,
    Y_BOUND,
    RENDER_SCALE,
    TASK_NAMES,
    LAYOUTS,
)


DEFAULT_VARIATIONS = (
    'agent.spawn',
    'agent.target_pad',
    'agent.visual',
)


# TODO: Re-enable targets to be sequences of pads instead of single pads
# TODO: Add walls to the environment, with the number of walls controlled by
# the variation space
class PinPad(gym.Env):
    def __init__(
        self,
        seed=None,
        init_value=None,
        render_mode='rgb_array',  # For backward compatibility; not used
        use_images=True,
    ):
        # Build variation space
        self.variation_space = self._build_variation_space()
        if init_value is not None:
            self.variation_space.set_init_value(init_value)

        # Other spaces
        self.observation_space = gym.spaces.Dict(
            {
                'image': gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(Y_BOUND * RENDER_SCALE, X_BOUND * RENDER_SCALE, 3),
                    dtype=np.uint8,
                ),
                'agent_position': gym.spaces.Box(
                    low=np.array([1.5, 1.5], dtype=np.float64),
                    high=np.array(
                        [X_BOUND - 1.5, Y_BOUND - 1.5], dtype=np.float64
                    ),
                    shape=(2,),
                    dtype=np.float64,
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float64,
        )

        # To be initialized in reset()
        self.task = None
        self.layout = None
        self.pads = None
        self.player = None
        self.target_pad = None
        self.use_images = use_images

    def _build_variation_space(self):
        return swm_spaces.Dict(
            {
                'agent': swm_spaces.Dict(
                    {
                        'spawn': swm_spaces.Box(
                            low=np.array([1.5, 1.5], dtype=np.float64),
                            high=np.array(
                                [X_BOUND - 1.5, Y_BOUND - 1.5],
                                dtype=np.float64,
                            ),
                            init_value=np.array(
                                [X_BOUND / 2, Y_BOUND / 2], dtype=np.float64
                            ),
                            shape=(2,),
                            dtype=np.float64,
                        ),
                        # The number of pads is dynamic based on the task,
                        # so we generate the index as a float in [0, 1) and then
                        # scale it to the number of pads before truncating it to an int
                        'target_pad': swm_spaces.Box(
                            low=0.0,
                            high=1.0,
                            init_value=0.0,
                            shape=(),
                            dtype=np.float64,
                        ),
                        'visual': swm_spaces.Discrete(
                            n=pinpad_assets.get_num_agent_images(),
                            start=0,
                            init_value=0,
                        ),
                    }
                ),
                'grid': swm_spaces.Dict(
                    {
                        'task': swm_spaces.Discrete(
                            n=len(TASK_NAMES),
                            start=0,
                            init_value=0,  # 0 = 'three', 5 = 'eight'
                        ),
                    }
                ),
            },
            sampling_order=['grid', 'agent'],
        )

    def _setup_layout(self, task):
        layout = LAYOUTS[task]
        self.layout = np.array(
            [list(line) for line in layout.split('\n')]
        ).T  # Transposes so that actions are (dx, dy)
        assert self.layout.shape == (X_BOUND, Y_BOUND), (
            f'Layout shape should be ({X_BOUND}, {Y_BOUND}), got {self.layout.shape}'
        )

    def _setup_pads(self):
        self.pads = sorted(
            list(set(self.layout.flatten().tolist()) - set('* #\n'))
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # Reset variation space
        options = options or {}
        swm_spaces.reset_variation_space(
            self.variation_space,
            seed,
            options,
            DEFAULT_VARIATIONS,
        )

        # Update task if it changed or if this is the first reset
        task_idx = int(self.variation_space['grid']['task'].value)
        new_task = TASK_NAMES[task_idx]
        if new_task != self.task or self.task is None:
            self.task = new_task
            self._setup_layout(self.task)
            self._setup_pads()

        # Set player position directly from variation space
        spawn_position = self.variation_space['agent']['spawn'].value
        self.player = tuple(spawn_position)

        # Set target pad from variation space using linear binning
        target_pad_value = float(
            self.variation_space['agent']['target_pad'].value
        )
        target_pad_idx = int(target_pad_value * len(self.pads))
        assert target_pad_idx >= 0 and target_pad_idx < len(self.pads), (
            f'Target pad index {target_pad_idx} is out of range for {len(self.pads)} pads'
        )
        self.target_pad = self.pads[target_pad_idx]
        self.goal_position = self._get_goal_position(self.target_pad)
        self.goal = self.render(player_position=self.goal_position)

        # Gets return values
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_obs(self):
        return {
            'image': self.render(),
            'agent_position': np.array(self.player, dtype=np.float64),
        }

    def step(self, action):
        # Moves player
        x = np.clip(self.player[0] + action[0], 1.5, X_BOUND - 1.5)
        y = np.clip(self.player[1] + action[1], 1.5, Y_BOUND - 1.5)
        tile = self.layout[int(x)][int(y)]
        if (
            tile != '#'
        ):  # TODO: Add linear interpolation in case of wall collision
            self.player = (float(x), float(y))

        # Makes observation
        agent_in_target_pad = self._agent_in_target_pad(
            self.player, self.target_pad
        )
        obs = self._get_obs()
        reward = 10.0 if agent_in_target_pad else 0.0
        terminated = agent_in_target_pad  # TODO: Maybe always set to false?
        truncated = False
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _get_goal_position(self, target_pad):
        target_cells = np.array(
            list(zip(*np.where(self.layout == target_pad))), dtype=np.float64
        )
        target_cell_centers = target_cells + 0.5
        center_cell = np.array([X_BOUND / 2, Y_BOUND / 2], dtype=np.float64)
        farthest_idx = np.argmax(
            np.linalg.norm(target_cell_centers - center_cell, axis=1)
        )
        farthest_from_center = target_cell_centers[farthest_idx]
        return farthest_from_center

    def _agent_in_target_pad(self, player, target_pad):
        # Gets all cells that overlap with the agent
        corner_deltas = np.array(
            [
                (-0.5, -0.5),
                (-0.5, 0.5),
                (0.5, -0.5),
                (0.5, 0.5),
            ],
            dtype=np.float64,
        )
        corner_positions = player + corner_deltas
        distinct_corner_positions = [
            tuple(pos)
            for pos in np.unique(corner_positions.astype(int), axis=0)
        ]

        # Gets all cells from the target pad
        target_cells = np.array(
            list(zip(*np.where(self.layout == target_pad))), dtype=np.float64
        )
        target_cells = [tuple(pos) for pos in target_cells.astype(int)]

        # Checks that the agent is entirely within the target pad
        for pos in distinct_corner_positions:
            if pos not in target_cells:
                return False
        return True

    def _get_info(self):
        info = {
            'goal_position': np.array(self.goal_position),
            'goal': self.goal,
        }
        return info

    def render(self, player_position=None):
        if player_position is None:
            player_position = self.player

        if self.use_images:
            return self._render_with_images(player_position)
        return self._render_with_colors(player_position)

    def _render_with_colors(self, player_position):
        """Original color-based rendering (pads as solid colors, agent as black dot)."""
        grid = np.zeros((X_BOUND, Y_BOUND, 3), np.uint8) + 255
        white = np.array([255, 255, 255])

        for (x, y), char in np.ndenumerate(self.layout):
            if char == '#':
                grid[x, y] = (192, 192, 192)  # Gray
            elif char in self.pads:
                color = np.array(COLORS[char])
                color = (
                    color
                    if self._agent_in_target_pad(player_position, char)
                    else (10 * color + 90 * white) / 100
                )
                grid[x, y] = color

        image = np.repeat(np.repeat(grid, RENDER_SCALE, 0), RENDER_SCALE, 1)
        image = image.transpose((1, 0, 2))

        image_pil = Image.fromarray(image, mode='RGB')
        draw = ImageDraw.Draw(image_pil)
        x, y = player_position
        half = 0.5 * pinpad_assets.AGENT_SIZE_FACTOR
        draw.rectangle(
            [
                (x - half) * RENDER_SCALE,
                (y - half) * RENDER_SCALE,
                (x + half) * RENDER_SCALE,
                (y + half) * RENDER_SCALE,
            ],
            fill=(0, 0, 0),
        )
        return np.asarray(image_pil)

    def _render_with_images(self, player_position):
        """Image-based rendering (food images for pads, animal image for agent)."""
        # Build base image: white background, gray walls
        height = Y_BOUND * RENDER_SCALE
        width = X_BOUND * RENDER_SCALE
        image = np.zeros((height, width, 3), dtype=np.uint8) + 255

        # Draw walls (gray)
        for (x, y), char in np.ndenumerate(self.layout):
            if char == '#':
                px_min, px_max = x * RENDER_SCALE, (x + 1) * RENDER_SCALE
                py_min, py_max = y * RENDER_SCALE, (y + 1) * RENDER_SCALE
                image[py_min:py_max, px_min:px_max] = (192, 192, 192)

        # Draw pads: group cells by pad char, get bbox, paste scaled image
        for pad_char in self.pads:
            cells = list(zip(*np.where(self.layout == pad_char)))
            if not cells:
                continue
            xs, ys = [c[0] for c in cells], [c[1] for c in cells]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            pad_w = (x_max - x_min + 1) * RENDER_SCALE
            pad_h = (y_max - y_min + 1) * RENDER_SCALE
            px_min, py_min = x_min * RENDER_SCALE, y_min * RENDER_SCALE

            pad_img = pinpad_assets.load_pad_image(pad_char, pad_w, pad_h)
            pinpad_assets._composite_rgba_onto_rgb(image, pad_img, px_min, py_min)

        # Draw agent (scaled up, centered on cell) - alpha composite so transparent areas show pads underneath
        agent_idx = int(self.variation_space['agent']['visual'].value)
        agent_size = int(RENDER_SCALE * pinpad_assets.AGENT_SIZE_FACTOR)
        agent_img = pinpad_assets.load_agent_image(agent_idx, agent_size)
        x, y = player_position
        center_px = int(x * RENDER_SCALE)
        center_py = int(y * RENDER_SCALE)
        px = center_px - agent_size // 2
        py = center_py - agent_size // 2
        px_clip = max(0, min(px, width - agent_size))
        py_clip = max(0, min(py, height - agent_size))
        src_x = max(0, -px)
        src_y = max(0, -py)
        dst_w = min(agent_size - src_x, width - px_clip)
        dst_h = min(agent_size - src_y, height - py_clip)
        if dst_w > 0 and dst_h > 0:
            agent_clip = agent_img[src_y : src_y + dst_h, src_x : src_x + dst_w]
            pinpad_assets._composite_rgba_onto_rgb(image, agent_clip, px_clip, py_clip)

        return image
