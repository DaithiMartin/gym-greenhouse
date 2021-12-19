import gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Callable
from numpy.typing import ArrayLike


class GreenhouseBaseEnv(gym.Env):
    """
    This is a Base Class for the Greenhouse Gym Environments.

    This base class should be extended to either a continuous and discrete form.
    The continuous form is used for Policy based methods and the discrete form is for value based methods.

    Starting episode will be 1 Day.
    Each step will be 1 hr.

    Current implementation based on: https://www.sciencedirect.com/science/article/pii/S0168169909001902

    Initial implementation based on:
    - single wall A-frame
    - 2 air exchanges per hour
    - evapotranspiration for large crop
    - Q_GRout calculated over 12 hour interval

    """
    metadata = {'render.modes': ['human']}

    def __init__(self):

        super(GreenhouseBaseEnv, self).__init__()

        # versions
        self.diurnal_swing: bool = True

        # simulation fields
        self.d_t: int = 1  # hours

        # cooling and heating with discrete steps
        # NOTE: action space needs to be calibrated to new physics
        self.num_heaters: int = 1
        self.action_max: float = 1e3  # Watts
        self.action_min: float = -1e3  # Watts
        self.action_space: gym.spaces = self.get_action_space()  # needs to be extended in specific implementation

        # FIXME: ASK GEORGE HOW TO ANNOTATE
        self.action_map: Union[dict, Callable] = self.get_action_map()  # needs to be extended in specific implementation

        # greenhouse dimensions, typical ratio if 3:1
        self.width: int = 10  # meters
        self.length: int = 30  # meters
        self.height: int = 4

        # state variables
        self.observation_space: gym.spaces = gym.spaces.Discrete(6)
        self.time: int = 0  # hour of the day
        self.outside_temp: ArrayLike = self.get_outside_temp()  # outside temp for each hour of the day
        self.solar_radiation: ArrayLike = self.get_radiative_heat()
        self.inside_temp: int = 22
        self.ideal_temp: int = 22
        self.temp_tolerance: int = 1

        # histories
        self.temp_history: list = []  # internal temp history for episode
        self.reward_history: list = []  # reward history for episode
        self.action_history: list = []  # action history for episode
        self.temp_change_history: list = []  # env temp change history for episode
        self.rad_temp_change_history: list = []  # radiative component of env temp change history for episode

    def get_action_space(self):
        """Extend to define a discrete or continuous action space"""

        raise NotImplementedError("Define an action space!")

        pass

    def get_action_map(self):
        """returns a function that maps from agent action to env action-space"""

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
        """Display episode visualization."""
        # internal vs external temperature
        x = np.arange(24)
        temp_external = self.outside_temp[:-1]  # exclude last time becuase thats hour 25
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
            temps = np.full(25, 25, dtype=int)  # len() = 25 because need post terminal info for last sarSa
        return temps

    def get_reward(self, action):
        # TODO: IMPLEMENT CLIFFING REWARD FUNCTION
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
        """Returns observation tuple"""

        time = self.time
        outside_temp = self.outside_temp[time]
        inside_temp = self.inside_temp
        ideal_temp = self.ideal_temp
        radiation = self.solar_radiation[time]
        in_tolerance = 1 if ideal_temp - self.temp_tolerance <= inside_temp <= ideal_temp + self.temp_tolerance else 0
        state = [time,
                 outside_temp,
                 inside_temp,
                 ideal_temp - inside_temp,
                 radiation,
                 in_tolerance
                 ]

        return np.array(state)

    def update_state(self, action):

        # agent takes action and environment reacts
        self.update_inside_temp(action)
        self.update_humidity(action)

        # update histories
        self.temp_history.append(self.inside_temp)
        self.action_history.append(self.action_map(action))

        # increment time
        self.time += 1

        # collect state
        high_tolerance = self.inside_temp + self.temp_tolerance
        low_tolerance = self.inside_temp - self.temp_tolerance
        nominal = 1 if high_tolerance >= self.inside_temp >= low_tolerance else 0
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

        # peak Q_GRout
        max_Q_GRout = 1025  # W m^-2
        day_length = 12
        day = np.sin(np.linspace(0, np.pi, day_length)) * max_Q_GRout
        night = np.zeros(13)

        Q_GRout = np.concatenate((day, night))

        return Q_GRout

    def update_inside_temp(self, action):
        """
        Energy Mass Balance for internal heat.

        Q_GRin + Q_heater = Q_IV + Q_glazing

        all units: W m^-2
        """
        # TODO: convert (W m^-2) to (J hr^-1 m^-2) ** this should be corrected but start by running unit tests
        # Q_heater
        num_heaters = self.num_heaters
        ground_surface = self.width * self.height
        heater_capacity = self.action_map(action)
        Q_heater = num_heaters * heater_capacity / ground_surface

        # Q_GRin
        tau_c = 0.9  # solar radiation transmittance of glazing material (dimensionless)
        rho_g = 0.5  # reflectance of solar radiation on ground (dimensionless)
        Q_GRout = self.solar_radiation[self.time]  # global outside radiation (W m^-2)
        Q_GRin = tau_c * (1 - rho_g) * Q_GRout

        # Q_IV
        # latent heat loss
        L = 2.5e6  # latent heat of vaporization (J kg^-1)
        E = (3e-4 * tau_c * Q_GRin + 0.0021) / 15 / 60  # evapotranspiration rate (kg m^-2 15min^-1) /15min /60 sec
        E = 0

        # sensible heat loss
        qv = 0.003  # ventilation rate (m^3 m^-2 s^-1)
        Cp = 1010  # specific heat of moist air (J kg^-1 K^-1)
        rho = 1.2  # specific mass of air  (kg dry air m^-3)
        T_in = self.inside_temp
        T_out = self.outside_temp[self.time]
        latent_loss = L * E
        sensible_loss = qv * Cp * rho * (T_in - T_out)
        Q_IV = latent_loss + sensible_loss

        # Q_glazing
        w = 2.2  # ratio of glazing surfaces to ground surface (dimensionless)
        k = 6.2  # heat transfer coefficient (W m^-2 C^-1)
        Q_glazing = k * w * (T_in - T_out)

        # first order Euler estimation
        H = 6.3  # average greenhouse height (m)

        d_Temp = 1 / (Cp * rho * H) * (Q_GRin + Q_heater - Q_IV - Q_glazing)  # deg C s^-1
        # d_Temp = 1 / (Cp * rho * H) * (Q_GRin + Q_heater - Q_glazing)

        d_Temp = d_Temp * 60  # deg C hr^-1

        self.inside_temp = self.inside_temp + d_Temp * self.d_t

        # add temp change to history
        self.temp_change_history.append(d_Temp)

        return None

    def update_humidity(self, action):
        pass

    def report(self):
        # TODO: COMPLETE ME WITH HISTORY REPORT
        """probably use a pandas dataframe for display"""
        pass


if __name__ == '__main__':
    env = GreenhouseBaseEnv()
