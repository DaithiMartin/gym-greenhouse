import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as plt


class GreenhouseBaseEnv(gym.Env):
    """
    This is a Base Class for the Greenhouse Gym Environments.

    This class is to be extended to a continuous and discrete form.
    The continuous form is used for Policy based methods and the discrete form is for value based methods.

    Starting episode will be 1 Day.
    Each step will be 1 hr.

    Assumptions:
    1. Heat loss from GH does not affect outside temperature.
    2. No heat loss through the ground

    """
    metadata = {'render.modes': ['human']}

    def __init__(self):

        super(GreenhouseBaseEnv, self).__init__()

        # versions
        self.diurnal_swing = True

        # cooling and heating with discrete steps
        self.action_min = -10
        self.action_max = 10
        self.action_space = self.get_action_space()
        self.action_map = self.get_action_map()

        # greenhouse dimensions, typical ratio if 3:1
        self.width = 10  # meters
        self.length = 30  # meters
        self.height = 4

        # state variables
        self.observation_space = gym.spaces.Discrete(6)
        self.time = 0  # hour of the day
        self.outside_temp = self.get_outside_temp()  # outside temp for each hour of the day
        self.solar_radiation = self.get_radiative_heat()
        self.inside_temp = 15
        self.ideal_temp = 22
        self.temp_tolerance = 1

        # histories
        self.temp_history = []  # internal temp history for episode
        self.reward_history = []    # reward history for episode
        self.action_history = []    # action history for episode
        self.temp_change_history = []   # env temp change history for episode
        self.rad_temp_change_history = []   # radiative component of env temp change history for episode

    def get_action_space(self):
        """Defines a discrete or continuous action space"""

        raise NotImplementedError("Define an action space!")

        pass

    def get_action_map(self):
        """returns a fucntion that maps """

        raise NotImplementedError("Define an action map!")

        pass

    def step(self, action):
        """
        Take a step in the environment.
        """

        state = self.update_state(action)
        reward = self.get_reward(action)
        done = False if self.time < 24 else True
        info = None

        return state, reward, done, info

    def reset(self):
        """Re-initialize Environment and return starting state"""

        self.__init__()
        state = self.get_state()

        return state

    def render(self, mode='human', report=False):
        # internal vs external temperature
        x = np.arange(24)
        temp_external = self.outside_temp[:-1]      # exclude last time becuase thats hour 25
        temp_internal = self.temp_history
        temp_ideal = np.full(24, self.ideal_temp)
        plt.plot(x, temp_external, label="External")
        plt.plot(x, temp_internal, 'o-', label="Internal")
        plt.fill_between(x, temp_ideal + self.temp_tolerance, temp_ideal - self.temp_tolerance,
                         label="Ideal", alpha=0.3, color='g')
        plt.xlabel("Time")
        plt.ylabel("Temperature")
        plt.legend()
        plt.show()
        if report:
            print(f"Actions")
            print(self.action_history)
            print(f"Rewards")
            print(self.reward_history)
            print(f"Temps")
            print(self.temp_history)

        return None

    def close(self):
        pass

    def get_outside_temp(self):

        if self.diurnal_swing:
            base = np.arange(6)
            base = 1.5 * base
            temps = np.array((base + 22, 22 + base[::-1], 22 - base, 22 - base[::-1])).flatten()
            temps = np.concatenate((temps, np.array([22])))
        else:
            # temps = np.full(24, np.random.randint(17, 22))
            temps = np.full(25, 25)     # len() = 25 because need post ternimal info for last sarSa
        return temps

    def get_reward(self, action):
        inside_temp = self.inside_temp
        ideal_temp = self.ideal_temp
        tolerance = self.temp_tolerance

        # calc current reward
        reward = -((inside_temp - ideal_temp) ** 2) * 100

        if ideal_temp + tolerance >= inside_temp >= ideal_temp - tolerance:
            reward += 1000

        # update history
        self.reward_history.append(reward)

        return reward

    def get_state(self):

        time = self.time
        outside_temp = self.outside_temp[time]
        inside_temp = self.inside_temp
        ideal_temp = self.ideal_temp
        radiation = self.solar_radiation[time]
        in_tolarance = 1 if ideal_temp - self.temp_tolerance <= inside_temp <= ideal_temp + self.temp_tolerance else 0
        state = [time,
                 outside_temp,
                 inside_temp,
                 ideal_temp - inside_temp,
                 radiation,
                 in_tolarance
                 ]

        return np.array(state)

    def update_state(self, action):

        # agent takes action
        self.inside_temp = self.inside_temp + self.action_map(action)

        # environment reacts
        self.update_temp(action)

        # update histories
        self.temp_history.append(self.inside_temp)
        self.action_history.append(self.action_map(action))

        # increment time
        self.time += 1

        # collect state
        nominal = 1 if self.inside_temp + self.temp_tolerance >= self.inside_temp >= self.inside_temp - self.temp_tolerance else 0
        state = [self.time,
                 self.outside_temp[self.time],
                 self.inside_temp,
                 self.ideal_temp - self.inside_temp,
                 self.solar_radiation[self.time],
                 nominal]

        return np.array(state)

    def update_conductive_flow(self):

        specific_heat = 1005.0  # J * kg^-1 K^-1, specific heat of "ambient" air
        air_volume = self.height * self.width * self.length  # m^3

        air_density = 1.225  # kg / m^3
        mass = air_volume * air_density  # kg

        # heat loss components
        area = 2 * self.width * self.height + 2 * self.length * self.height + self.width * self.height
        U = 0.5  # typical 2 layer window value, U = 1/R, will need to be updated
        T_outside = self.outside_temp[self.time]
        T_inside = self.inside_temp

        # heat lost by system through conduction
        dQ = U * area * (T_outside - T_inside)  # watts lost to environment
        time = self.time

        # convert watts to jules in 1 hour
        # watt is a jule/sec
        dQ = dQ * 60 * 60  # jules lost to environment

        # calc new green house temp after heat loss
        temp_change = (1 / specific_heat) * dQ / mass

        new_temp = T_inside + temp_change

        self.inside_temp = new_temp

        return temp_change

    @staticmethod
    def get_radiative_heat():
        swing = np.arange(6)
        swing = 1.5 * swing
        base_line = 0
        radiation = np.array(
            (swing + base_line, base_line + swing[::-1], np.full(6, base_line), np.full(6, base_line))).flatten()

        radiation = np.concatenate((radiation, np.zeros(1)))

        return radiation

    def update_radiative_flow(self):
        # FIXME: THIS IS FUNCTIONING BUT NEEDS SIGNIFICANT WORK TO BE MORE REPRESENTATIVE OF REALITY
        radation = self.solar_radiation[self.time]
        specific_heat = 1005.0  # J * kg^-1 K^-1, specific heat of "ambient" air
        air_volume = self.height * self.width * self.length  # m^3

        air_density = 1.225  # kg / m^3
        mass = air_volume * air_density  # kg
        factor = 1e6
        dQ = radation * factor
        temp_change = (1 / specific_heat) * dQ / mass

        self.rad_temp_change_history.append(temp_change)

        # update internal representation of temperature
        T_inside = self.inside_temp

        new_temp = T_inside + temp_change

        self.inside_temp = new_temp

        return temp_change

    def update_temp(self, action):

        radiation_change = self.update_radiative_flow()
        conductive_change = self.update_conductive_flow()

        temp_change = radiation_change + conductive_change

        self.temp_change_history.append(temp_change)

        return None

    def report(self):
        # TODO: COMPLETE ME WITH HISTORY REPORT
        """probably use a pandas dataframe for display"""
        pass


if __name__ == '__main__':
    env = GreenhouseBaseEnv()

