"""Discrete greenhouse environment"""
from typing import Callable, Any
import gym
import numpy as np
from numpy import linspace
from gym_greenhouse.envs.greenhouse_base import GreenhouseBaseEnv

# Hyper parameters
# -------------------------------------------------------------------------------#
NUM_ACTIONS = 41


# -------------------------------------------------------------------------------#


class GreenhouseDiscreteEnv(GreenhouseBaseEnv):
    """Greenhouse simulator with discrete action space"""

    def __init__(self):
        super().__init__()
        self.action_space: Any = self.get_action_space()

        self.action_dict: dict = self.generate_action_dict()
        self.action_map: Callable = self.get_action_map()

    @staticmethod
    def get_action_space() -> Any:
        """
        Factory for generating correct action space.

        Returns: gym.spaces object for discrete action space
        """

        action_space = gym.spaces.Discrete(NUM_ACTIONS)

        return action_space

    def generate_action_dict(self) -> dict:
        """
        Generates dict that maps agent action index to environment action.

        Returns: dictionary mapping agent action to env action.
        """
        num_actions = self.action_space.n
        action_range = linspace(self.action_min, self.action_max, num_actions)
        index_range = range(num_actions)
        action_dict = {}
        for index, action_val in zip(index_range, action_range):
            action_dict[index] = action_val

        return action_dict

    def get_action_map(self) -> Callable:
        """
        Factory for making action map callable object.

        Returns: function that calls the actin dict.
        """

        def action_map(action):
            return self.action_dict[action]

        return action_map


if __name__ == "__main__":
    env = GreenhouseDiscreteEnv()
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")

    observation = env.reset()
    print(f"Initial Observation: {observation}")
    # actions = [9, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 7, 7, 9, 10, 10, 10, 10, 8, 7, 7, 5]
    # actions = np.zeros(80, dtype=int).tolist()  # apply cooling the whole time, to test if action space is large enough
    # wat_action = env.action_map(0)
    for t in range(80):
        # action = env.action_space.sample()    # random action
        agent_action = env.action_space.n // 2  # take no action
        # agent_action = actions[t]  # specific trajectory
        observation, reward, done, info = env.step(agent_action)
        # print(f"Observation {t + 1}: {observation}")
        if done:
            print(f"Episode finished after {t + 1} time-steps")
            break

    env.render()

    print("complete")
