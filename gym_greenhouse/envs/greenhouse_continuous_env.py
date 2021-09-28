
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt
from gym_greenhouse.envs.greenhouse_base import GreenhouseBaseEnv


#
#
# class GreenhouseContinuousEnv(GreenhouseBaseEnv):
#     """
#     Greenhouse environment with continuous action space.
#     """
#     metadata = {'render.modes': ['human']}
#
#     def __init__(self):
#
#         super(GreenhouseContinuousEnv, self).__init__(action_space="continuous")
#
#         # versions
#         self.diurnal_swing = True
#
#         # 5 degrees cooling or 5 degrees heating
#         self.action_space = gym.spaces.Box(
#             np.array([-5]).astype(np.float32),
#             np.array([5]).astype(np.float32),
#         )
#
#         # greenhouse dimensions, typical ratio if 3:1
#         self.width = 10  # meters
#         self.length = 30  # meters
#         self.height = 4
#
#         # state variables
#         self.observation_space = gym.spaces.Discrete(5)
#         # self.reward = 0     # try a cumulative reward
#         self.time = 0  # hour of the day
#         self.outside_temp = self.get_outside_temp()  # outside temp for each hour of the day
#         # self.inside_temp = np.random.randint(0, 30)  # initial inside temperature
#         self.inside_temp = 15
#         self.ideal_temp = 22
#         self.temp_tolerance = 1
#
#         # histories
#         self.temp_history = np.zeros(24)  # internal temp history
#         self.reward_history = []  # reward history for cumulative reward at terminal state
#         self.action_history = np.zeros(24)
#         self.temp_change_history = np.zeros(24)
#
#
#     def step(self, action):
#         """
#         Take a step in the environment.
#         """
#
#         state = self.update_state(action)
#         reward = self.get_reward(action)
#         done = False if self.time < 24 else True
#         info = None
#
#         return state, reward, done, info
#
#     def reset(self):
#
#         self.__init__()
#         state = self.get_state()
#
#         return state
#
#     def render(self, mode='human', report=False):
#         # internal vs external temperature
#         x = np.arange(24)
#         temp_external = self.outside_temp[:-1]      # exclude last time becuase thats hour 25
#         temp_internal = self.temp_history
#         temp_ideal = np.full(24, self.ideal_temp)
#         plt.plot(x, temp_external, label="External")
#         plt.plot(x, temp_internal, 'o-', label="Internal")
#         plt.fill_between(x, temp_ideal + self.temp_tolerance, temp_ideal - self.temp_tolerance,
#                          label="Ideal", alpha=0.3, color='g')
#         plt.xlabel("Time")
#         plt.ylabel("Temperature")
#         plt.legend()
#         plt.show()
#         if report:
#             print(f"Actions")
#             print(self.action_history)
#             print(f"Rewards")
#             print(self.reward_history)
#             print(f"Temps")
#             print(self.temp_history)
#
#         return None
#
#     def close(self):
#         pass
#
#     def get_outside_temp(self):
#
#         if self.diurnal_swing:
#             base = np.arange(6)
#             base = 1.5 * base
#             temps = np.array((base + 22, 22 + base[::-1], 22 - base, 22 - base[::-1])).flatten()
#             temps = np.concatenate((temps, np.array([22])))
#         else:
#             # temps = np.full(24, np.random.randint(17, 22))
#             temps = np.full(25, 25)     # len() = 25 because need post ternimal info for last sarSa
#         return temps
#
#     def get_reward(self, action):
#         """
#
#         :param action: action taken by agent, [heating, cooling]
#         :return: reward
#         """
#         inside_temp = self.inside_temp
#         ideal_temp = self.ideal_temp
#         tolerance = self.temp_tolerance
#
#         # calc current reward
#         reward = -((inside_temp - ideal_temp) ** 2) * 100
#
#         if ideal_temp + tolerance >= inside_temp >= ideal_temp - tolerance:
#             reward += 1000
#
#         # update history
#         self.reward_history.append(reward)
#
#         return reward
#
#     def get_state(self):
#         # state = (self.time, self.outside_temp[self.time], self.inside_temp, self.ideal_temp)
#         # state = np.array((self.time, self.outside_temp[self.time], self.inside_temp, self.ideal_temp - self.inside_temp))
#         time = self.time
#         outside_temp = self.outside_temp[time]
#         inside_temp = self.inside_temp
#         ideal_temp = self.ideal_temp
#         in_tolarance = 1 if ideal_temp - self.temp_tolerance <= inside_temp <= ideal_temp + self.temp_tolerance else 0
#         state = [time,
#                  outside_temp,
#                  inside_temp,
#                  ideal_temp - inside_temp,
#                  in_tolarance
#                  ]
#
#         return np.array(state)
#
#     def update_state(self, action):
#
#         # agent tackes action
#         self.inside_temp = self.inside_temp + (action.item() * 5)
#
#         # environment reacts
#         self.update_temp(action)
#
#         # update histories
#         self.temp_history[self.time] = self.inside_temp
#         self.action_history[self.time] = action.item() * 5
#
#         # increment time
#         self.time += 1
#
#         # collect state
#         nominal = 1 if self.inside_temp + self.temp_tolerance >= self.inside_temp >= self.inside_temp - self.temp_tolerance else 0
#         state = [self.time,
#                  self.outside_temp[self.time],
#                  self.inside_temp,
#                  self.ideal_temp - self.inside_temp,
#                  nominal]
#
#         return np.array(state)
#
#     def update_temp(self, action):
#         specific_heat = 1005.0  # J * kg^-1 K^-1, specific heat of "ambient" air
#         air_volume = self.height * self.width * self.length  # m^3
#
#         air_density = 1.225  # kg / m^3
#         mass = air_volume * air_density  # kg
#
#         # heat loss components
#         area = 2 * self.width * self.height + 2 * self.length * self.height + self.width * self.height
#         U = 0.5  # typical 2 layer window value, U = 1/R, will need to be updated
#         T_outside = self.outside_temp[self.time]
#         T_inside = self.inside_temp
#
#         # heat loss through conduction
#         dQ = U * area * (T_outside - T_inside)  # watts lost to environment
#
#         # convert watts to jules in 1 hour
#         # watt is a jule/sec
#         dQ = dQ * 60 * 60  # jules lost to environment
#
#         # calc new green house temp after heat loss
#         temp_change = (1 / specific_heat) * dQ / mass
#
#         self.temp_change_history[self.time] = temp_change
#
#         new_temp = T_inside + temp_change
#
#         self.inside_temp = new_temp
#
#         return None
#
#
# if __name__ == '__main__':
#     env = GreenhouseContinuousEnv()
#
#     observation = env.reset()
#     print(f"Initial Observation: {observation}")
#     for t in range(50):
#         action = env.action_space.sample()
#         observation, reward, done, info = env.step(action)
#         print(f"Observation {t + 1}: {observation}")
#         if done:
#             print("Episode finished after {} time-steps".format(t + 1))
#             break
#
#     print(f"temp history{env.temp_history}, Length: {len(env.temp_history)}")
#     print(f"reward history {env.reward_history}, Length: {len(env.reward_history)}")
#
#     x = np.arange(24)
#     temp_external = env.outside_temp[:-1]
#     temp_internal = env.temp_history
#     plt.plot(x, temp_external, label="External")
#     plt.plot(x, temp_internal, label="Internal")
#     plt.legend()
#     plt.show()

class GreenhouseContinuousEnv(GreenhouseBaseEnv):
    def __init__(self):
        super(GreenhouseContinuousEnv, self).__init__()
        self.action_space = self.get_action_space()
        self.action_map = self.get_action_map()

    def get_action_space(self):
        """Defines a continuous action space."""
        action_space = gym.spaces.Box(
            np.array([-self.action_min]).astype(np.float32),
            np.array([self.action_max]).astype(np.float32),
        )
        return action_space

    def get_action_map(self):
        """produces a function that maps from agent action space to environment action space

        Currently assumes that self.action_max and self.action_min are equidistant from no-action
        """

        def action_map(agent_action):
            action = agent_action.item() * self.action_max
            return action

        return action_map


if __name__ == '__main__':
    env = GreenhouseContinuousEnv()
    observation = env.reset()
    print(f"Initial Observation: {observation}")
    for t in range(50):
        action = env.action_space.sample()
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