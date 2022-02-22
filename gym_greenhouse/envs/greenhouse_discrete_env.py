"""Discrete greenhouse environment"""
from typing import Callable, Any
import gym
from numpy import linspace
from gym_greenhouse.envs.greenhouse_base import GreenhouseBaseEnv


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

        num_actions = 21
        action_space = gym.spaces.Discrete(num_actions)

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
        # TODO: SHOW FACTORY TO GEORGE
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
    for t in range(50):
        # action = env.action_space.sample()    # random action
        agent_action = env.action_space.n // 2  # take no action
        # action = actions[t]                   # specific trajectory
        observation, reward, done, info = env.step(agent_action)
        print(f"Observation {t + 1}: {observation}")
        if done:
            print(f"Episode finished after {t + 1} time-steps")
            break

    # TODO: REPLACE WITH REPORT FUNCTION IN BASE CLASS
    print(f"temp history: {env.temp_history}, Length: {len(env.temp_history)}")
    print(f"reward history: {env.reward_history}, Length: {len(env.reward_history)}")
    print(
        f"temp change history: {env.temp_change_history}, Length: {len(env.temp_change_history)}"
    )
    print(f"Action history: {env.action_history}")
    print(f"rad temp change history: {env.rad_temp_change_history}")

    env.render()
