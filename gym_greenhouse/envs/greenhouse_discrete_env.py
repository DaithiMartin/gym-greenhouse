
import gym
from gym_greenhouse.envs.greenhouse_base import GreenhouseBaseEnv


class GreenhouseDiscreteEnv(GreenhouseBaseEnv):

    def __init__(self):
        super(GreenhouseDiscreteEnv, self).__init__()
        self.action_space = self.get_action_space()

        self.action_dict = self.generate_action_dict()
        self.action_map = self.get_action_map()

    def get_action_space(self):

        num_actions = self.action_max - self.action_min + 1
        action_space = gym.spaces.Discrete(num_actions)

        return action_space

    def generate_action_dict(self):

        action_range = range(self.action_min, self.action_max + 1)
        num_actions = self.action_space.n
        index_range = range(num_actions)
        action_dict = {}
        for index, action in zip(index_range, action_range):
            action_dict[index] = action

        return action_dict

    def get_action_map(self):

        def action_map(action):
            return self.action_dict[action]

        return action_map


if __name__ == '__main__':
    env = GreenhouseDiscreteEnv()

    observation = env.reset()
    print(f"Initial Observation: {observation}")
    # actions = [9, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 7, 7, 9, 10, 10, 10, 10, 8, 7, 7, 5]
    for t in range(50):
        action = env.action_space.sample()
        # action = actions[t]
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
