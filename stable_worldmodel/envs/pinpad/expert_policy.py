import numpy as np
from stable_worldmodel.policy import BasePolicy


class ExpertPolicyDiscrete(BasePolicy):
    """Expert policy for the PinPadDiscrete environment."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.type = 'expert'

    def compute_action(self, agent_position, target_position):
        dx, dy = (target_position - agent_position).tolist()
        if abs(dx) + abs(dy):
            # Gets directions we need to move in (in the transposed space)
            possible_actions = []
            if abs(dx):
                if dx > 0:
                    possible_actions.append(3)  # right
                else:
                    possible_actions.append(4)  # left
            if abs(dy):
                if dy > 0:
                    possible_actions.append(1)  # up
                else:
                    possible_actions.append(2)  # down

            # Alternates between horizontal and vertical moves
            if len(possible_actions) == 2:
                action = possible_actions[(abs(dx) + abs(dy)) % 2]
            else:
                action = possible_actions[0]
        else:
            action = 0
        return action

    def get_action(self, info_dict, **kwargs):
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert 'agent_position' in info_dict, 'Agent position must be provided in info_dict'
        assert 'target_position' in info_dict, 'Target position must be provided in info_dict'

        # Check if environment is vectorized
        base_env = self.env.unwrapped
        if hasattr(base_env, 'envs'):
            envs = [e.unwrapped for e in base_env.envs]
            is_vectorized = True
        else:
            envs = [base_env]
            is_vectorized = False

        # Computes actions for each environment
        actions = np.zeros(len(envs), dtype=np.int64)
        for i, env in enumerate(envs):
            if is_vectorized:
                agent_position = np.asarray(
                    info_dict['agent_position'][i], dtype=np.int64
                ).squeeze()
                target_position = np.asarray(
                    info_dict['target_position'][i], dtype=np.int64
                ).squeeze()
            else:
                agent_position = np.asarray(
                    info_dict['agent_position'], dtype=np.int64
                ).squeeze()
                target_position = np.asarray(
                    info_dict['target_position'], dtype=np.int64
                ).squeeze()

            actions[i] = self.compute_action(agent_position, target_position)

        return actions if is_vectorized else actions[0]
