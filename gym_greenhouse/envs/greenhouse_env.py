import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt


class GreenhouseEnv(gym.Env):
    """
    Starting episode will be 1 Day.
    Each step will be 1 hr.

    Assumptions:
    1. Heat loss from GH does not affect outside temperature.
    2. No heat loss through the ground

    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # dim_1 = heat 0-5, dim_2 = cooling 0-5
        self.action_space = gym.spaces.MultiDiscrete([5, 5])

        # greenhouse dimensions, typical ratio if 3:1
        self.width = 10  # meters
        self.length = 30  # meters
        self.height = 4

        # state variables
        self.observation_space = gym.spaces.Discrete(4)
        self.time = 0  # hour of the day
        self.outside_temp = self.get_outside_temp()  # outside temp for each hour of the day
        self.inside_temp = 22  # initial inside temperature
        self.ideal_temp = 22

        # histories
        self.temp_history = np.zeros(24)  # internal temp history
        self.reward_history = np.zeros(24)  # reward history for cumulative reward at terminal state

    def step(self, action):
        """
        Take a step in the environment.
        """

        state = self.get_state(action)
        reward = self.get_reward(action)
        done = False if self.time < 23 else True
        info = None

        # # terminal state, calc reward and return done
        # if done:
        #     reward = -np.sum(np.abs(self.temp_history - self.ideal_temp))
        #     reward = reward + np.sum(self.reward_history)
        #     self.reward_history[self.time] = reward
        #     return state, reward, done, info

        # increment time
        self.time += 1

        return state, reward, done, info

    def reset(self):
        self.time = 0  # hour of the day
        self.outside_temp = self.get_outside_temp()  # outside temp for each hour of the day
        self.inside_temp = 22  # initial inside temperature
        self.ideal_temp = 22

        # histories
        self.temp_history = np.zeros(24)  # internal temp history
        self.reward_history = np.zeros(24)  # reward history for cumulative reward at terminal state

        state = (self.time, self.outside_temp[self.time], self.inside_temp, self.ideal_temp)

        return state

    def render(self, mode='human'):
        # internal vs external temperature
        x = np.arange(24)
        temp_external = self.outside_temp
        temp_internal = self.temp_history
        plt.plot(x, temp_external, label="External")
        plt.plot(x, temp_internal, label="Internal")
        plt.legend()
        plt.show()

        return None

    def close(self):
        pass

    @staticmethod
    def get_outside_temp():
        base = np.arange(6)
        base = 1.5 * base
        temps = np.array((base + 22, 22 + base[::-1], 22 - base, 22 - base[::-1])).flatten()

        return temps

    def get_reward(self, action):
        """
        currently reward is directly proportional to the action.
        :param action: action taken by agent, [heating, cooling]
        :return: reward
        """
        # calc current reward
        reward = -(np.sum(action) / 100) - np.abs(self.inside_temp - self.ideal_temp)

        # update history
        self.reward_history[self.time] = reward

        return reward

    def get_state(self, action):
        # split actions
        heat_input = action[0]
        cooling_input = action[1]

        # generate state
        time = self.time
        outside_temp = self.outside_temp[self.time]
        inside_temp = self.get_new_temp(heat_input, cooling_input)
        ideal_temp = self.ideal_temp

        state = (time, outside_temp, inside_temp, ideal_temp)

        # update temp history
        self.temp_history[self.time] = inside_temp

        return state

    def get_new_temp(self, heat_input, cooling_input):
        specific_heat = 1005.0  # J * kg^-1 K^-1, specific heat of "ambient" air
        air_volume = self.height * self.width * self.length  # m^3

        air_density = 1.225  # kg / m^3
        mass = air_volume * air_density  # kg

        # heat loss components
        area = 2 * self.width * self.height + 2 * self.length * self.height + self.width * self.height
        U = 2  # typical 2 layer window value, U = 1/R, will need to be updated
        T_outside = self.outside_temp[self.time]
        T_indide = self.inside_temp

        # heat loss through conduction
        dQ = U * area * (T_indide - T_outside)  # watts lost to environment

        # convert watts to jules
        dQ = dQ * 60 * 60  # jules lost to environment

        # calc new green house temp after heat loss
        temp_change = (1 / specific_heat) * dQ / mass

        new_temp = T_indide - temp_change

        # calc new greenhouse temp after agent actions
        new_temp = new_temp + heat_input - cooling_input

        # update internal temperature
        self.inside_temp = new_temp

        return new_temp


if __name__ == '__main__':
    env = GreenhouseEnv()

    observation = env.reset()
    print(f"Initial Observation: {observation}")
    for t in range(50):
        # action = env.action_space.sample()
        action = [0, 0]
        observation, reward, done, info = env.step(action)
        print(f"Observation {t + 1}: {observation}")
        if done:
            print("Episode finished after {} time-steps".format(t + 1))
            break

    print(f"temp history{env.temp_history}, Length: {len(env.temp_history)}")
    print(f"reward history {env.reward_history}, Length: {len(env.reward_history)}")

    x = np.arange(24)
    temp_external = env.outside_temp
    temp_internal = env.temp_history
    plt.plot(x, temp_external, label="External")
    plt.plot(x, temp_internal, label="Internal")
    plt.legend()
    plt.show()
