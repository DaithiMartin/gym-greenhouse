
import gym
import numpy as np
from gym_greenhouse.envs.greenhouse_base import GreenhouseBaseEnv


class GreenhouseContinuousEnv(GreenhouseBaseEnv):
    def __init__(self):
        super(GreenhouseContinuousEnv, self).__init__()
        self.action_space = self.get_action_space()
        self.action_map = self.get_action_map()

    def get_action_space(self):
        """Defines a continuous action space."""
        action_space = gym.spaces.Box(
            np.array([self.action_min]).astype(np.float32),
            np.array([self.action_max]).astype(np.float32),
        )
        return action_space

    def get_action_map(self):
        """produces a function that maps from agent action space to environment action space

        Currently assumes that self.action_max and self.action_min are equidistant from no-action
        """
        # TODO: SHOW FACTORY TO GEORGE
        def action_map(agent_action):
            action = agent_action.item() * self.action_max
            return action

        return action_map


if __name__ == '__main__':
    env = GreenhouseContinuousEnv()
    observation = env.reset()
    print(f"Initial Observation: {observation}")
    for t in range(50):
        # action = env.action_space.sample()
        action = np.array(0.0)
        observation, reward, done, info = env.step(action)
        print(f"Observation {t + 1}: {observation}")
        if done:
            print("Episode finished after {} time-steps".format(t + 1))
            break

    print(f"temp history: {env.temp_history}, Length: {len(env.temp_history)}")
    print(f"reward history: {env.reward_history}, Length: {len(env.reward_history)}")
    print(f"temp change history: {env.temp_change_history}, Length: {len(env.temp_change_history)}")
    print(f"Action history: {env.action_history}")
    print(f"rad temp change history: {env.rad_temp_change_history}")

    env.render()