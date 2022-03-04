import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Union, Callable, Any, List, TypeVar, Generic
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

# Hyper parameters
# -------------------------------------------------------------------------------#
# Environment Type
ACTION_MAX: float = 1e5  # watts
ACTION_MIN: float = -1e5  # watts
DIURNAL_SWING: bool = True

# Greenhouse simulation
SPECIFIC_HEAT: float = 1005.0  # J * kg^-1 K^-1, specific heat of "ambient" air
AIR_DENSITY: float = 1.225  # kg / m^3

# -------------------------------------------------------------------------------#


# Typer parameters
# -------------------------------------------------------------------------------#
T = TypeVar("T")


# -------------------------------------------------------------------------------#


class GreenhouseBaseEnv(gym.Env, Generic[T]):
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
        self.diurnal_swing: bool = DIURNAL_SWING

        # simulation fields
        self.d_t: int = 1  # hours

        # cooling and heating with discrete steps
        # NOTE: action space needs to be calibrated to new physics
        self.num_heaters: int = 1
        self.action_max: float = ACTION_MAX  # Watts
        self.action_min: float = ACTION_MIN  # Watts
        self.action_space: gym.spaces = self.get_action_space()  # needs to be extended

        # FIXME: make both callables
        self.action_map: Union[dict, Callable] = self.get_action_map()  # needs to be extended

        # greenhouse dimensions, typical ratio if 3:1
        self.width: int = 10  # meters
        self.length: int = 30  # meters
        self.height: float = 6.3  # meters

        # state variables
        self.observation_space: gym.spaces = gym.spaces.Discrete(6)
        self.time: int = 0  # hour of the day
        self.outside_temp: ArrayLike = self.set_outside_temp()  # set outside temp for episode
        self.solar_radiation: ArrayLike = self.set_radiative_heat()     # set radiative radiation for episode
        self.inside_temp: int = 22      # starting inside temp
        self.ideal_temp: int = 22       # ideal inside temp
        self.temp_tolerance: int = 1    # +/- tolerance for ideal temp
        self.outside_rh_humid: ArrayLike = self.set_outside_rh()    # set relative humidity for episode
        self.inside_abs_humid: float = self.map_rel_to_abs_humid(self.inside_temp, 0.2)   # initial inside temp
        self.ideal_rel_humid: float = 0.1   # ideal relative humidity
        self.humid_tolerance: float = 0.05  # +/- tolerance for ideal humidity

        # histories
        self.final_temp_history: List[float] = []  # internal temp history for episode
        self.reward_history: List[float] = []  # reward history for episode
        self.action_history: List[T] = []  # action history for episode
        self.temp_change_history: list = []  # env temp change history for episode
        self.rad_temp_change_history: list = []  # radiative component of env temp change history for episode
        self.start_temp_hist: list = []
        self.initial_rel_humid_history: list = []
        self.final_rel_humid_history: list = []

        # dataframe formatted report
        self.episode_report = None

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

    def render(self,
               mode: str = 'human',
               report: bool = True):

        """Display episode visualization."""

        # internal vs external temperature
        x = np.arange(24)
        temp_external = self.outside_temp[:-1]  # exclude last time because that's hour 25
        temp_internal = self.final_temp_history
        temp_ideal = np.full(24, self.ideal_temp)
        plt.plot(x, temp_external, label="External")
        plt.plot(x, temp_internal, 'o-', label="Internal")
        plt.fill_between(x, temp_ideal + self.temp_tolerance, temp_ideal - self.temp_tolerance,
                         label="Ideal", alpha=0.3, color='g')
        plt.title("Temperature")
        plt.xlabel("Time")
        plt.ylabel("Degree C")
        plt.legend()
        plt.show()

        # internal humidity vs external humidity
        humid_external = self.outside_rh_humid[:-1]
        humid_internal = self.final_rel_humid_history
        humid_ideal = np.full(24, self.ideal_rel_humid)
        plt.plot(x, humid_external, label="External")
        plt.plot(x, humid_internal, 'o-', label="Internal")
        plt.fill_between(x, humid_ideal + self.humid_tolerance, humid_ideal - self.humid_tolerance,
                         label="Ideal", alpha=0.3, color='y')
        plt.title("Humidity")
        plt.xlabel("Time")
        plt.ylabel("Relative Humidity")
        plt.legend()
        plt.show()

        if report:
            self.episode_report = self.report()

        return None

    def close(self):
        pass

    def set_outside_temp(self):

        if self.diurnal_swing:
            base = np.arange(6)
            base = 1.5 * base
            temps = np.array((base + 22, 22 + base[::-1], 22 - base, 22 - base[::-1])).flatten()
            temps = np.concatenate((temps, np.array([22])))
        else:
            # temps = np.full(24, np.random.randint(17, 22))
            temps = np.full(25, 26, dtype=int)  # len() = 25 because need post terminal info for last sarSa
        return temps

    @staticmethod
    def set_outside_rh():

        rel_humid = np.full(25, 0.2)
        return rel_humid

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
        """Returns observation tuple of current state."""

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
        """Updates internal state"""
        # log initial temperature
        self.start_temp_hist.append(self.inside_temp)
        self.initial_rel_humid_history.append(self.map_abs_to_rel_humid(self.inside_temp, self.inside_abs_humid))

        # agent takes action and environment reacts
        new_temp, new_humid = self.get_temp_humid(action)

        # update instance state
        self.inside_temp = new_temp
        self.inside_abs_humid = new_humid

        # update histories
        self.final_temp_history.append(self.inside_temp)
        self.final_rel_humid_history.append(self.map_abs_to_rel_humid(self.inside_temp, self.inside_abs_humid))
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
        """DEPRECIATED"""

        air_volume = self.height * self.width * self.length  # m^3

        mass = air_volume * AIR_DENSITY  # kg

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
        temp_change = (1 / SPECIFIC_HEAT) * dQ / mass

        new_temp = T_inside + temp_change

        self.inside_temp = new_temp

        return temp_change

    @staticmethod
    def set_radiative_heat():

        # peak Q_GRout
        max_Q_GRout = 1025  # W m^-2
        day_length = 12
        day = np.sin(np.linspace(0, np.pi, day_length)) * max_Q_GRout
        night = np.zeros(13)

        Q_GRout = np.concatenate((day, night))

        return Q_GRout

    @staticmethod
    def map_rel_to_abs_humid(temp, rel_humid):
        """Maps relative humidity -> absolute humidity"""
        abs_humid = (6.112 * np.exp((17.67 * temp) / (temp + 243.5)) * rel_humid * 2.1674)
        abs_humid = abs_humid / (273.15 + temp)

        return abs_humid

    @staticmethod
    def map_abs_to_rel_humid(temp, abs_humid):
        """Masp absolute humidity -> relative humidity"""
        rel_humid = abs_humid * (273.15 + temp)
        rel_humid = rel_humid / (6.112 * np.exp((17.67 * temp) / (temp + 243.5)) * 2.1674)

        return rel_humid

    def get_q_heater(self,
                     action: tuple):

        ground_surface = self.width * self.length
        heater_capacity = self.action_map(action)
        Q_heater = self.num_heaters * heater_capacity / ground_surface

        return Q_heater

    def energy_bal(self, t, T_in, *args):
        """RHS of temperature energy balance equation"""

        # constant parameters
        Cp = 1010  # specific heat of moist air (J kg^-1 K^-1)
        rho = 1.2  # specific mass of air  (kg dry air m^-3)
        H = self.height  # Note that the paper calls for avg height but does not describe how that is calculated(m)
        L = 2.5e6  # latent heat of vaporization (J kg^-1)
        qv = 0.003  # ventilation rate (m^3 m^-2 s^-1)
        # TODO: DETERMINE HOW TO MAKE THIS DYNAMIC WITH DIFFERENT GH ARCHITECTURE
        w = 2.1  # ration of glazing to floor area
        k = 6.2  # heat transfer coefficient (W m^-2 C^-1)

        # non constant parameters
        Q_GRin = args[0]
        Q_Heat = args[1]
        E = args[2]
        T_out = args[3]

        # components
        Q_latent = L * E
        Q_sensible = qv * Cp * rho * (T_in - T_out)
        Q_glaze = k * w * (T_in - T_out)

        d_Temp = 1 / (Cp * rho * H) * (Q_GRin + Q_Heat - Q_latent - Q_sensible - Q_glaze)

        return d_Temp

    def water_balance(self, t, W_in, *args):
        """RHS of humidity mass balance equation"""

        # constant parameters
        H = self.height  # Note that the paper calls for avg height but does not describe how that is calculated(m)
        rho = 1.2  # specific mass of air  (kg dry air m^-3)
        qv = 0.003  # ventilation rate (m^3 m^-2 s^-1)

        # non-constant parameters
        E = args[0]
        W_out = args[1]

        d_W = 1 / (H * rho) * (E - (W_in - W_out) * qv * rho)

        return d_W

    def get_temp_humid(self, action):
        """
        Energy and Mass Balance for internal heat and humidity.

        Heat, all units: W m^-2
        Q_GRin + Q_heater = Q_IV + Q_glazing

        Water, absolute humidity in kg water kg^-1 dry air
        W_in * q_v * rho = W_out * q_v * rho + E

        **note 1 W == J s^-1
        """

        # Q_heater
        Q_heater = self.get_q_heater(action)

        # Q_GRin
        tau_c = 0.9  # solar radiation transmittance of glazing material (dimensionless)
        rho_g = 0.5  # reflectance of solar radiation on ground (dimensionless)
        Q_GRout = self.solar_radiation[self.time]  # global outside radiation (W m^-2)
        Q_GRin = tau_c * (1 - rho_g) * Q_GRout

        # latent heat
        # energy consumed in the phase transition of water, in this simulation: evapotranspiration
        E = (3e-4 * tau_c * Q_GRin + 0.0021) / 15 / 60  # evapotranspiration rate (kg m^-2 15min^-1) /15min /60 sec
        # E = 0  # no plants

        # ode time period, 60 sec min^-1 * 60 min hr^-1
        t_span = (0, 60 * 60)

        # energy ode solver
        T_out = self.outside_temp[self.time]
        energy_args = (Q_GRin, Q_heater, E, T_out)
        energy_sol = solve_ivp(self.energy_bal, t_span, [self.inside_temp], args=energy_args)
        new_temp = energy_sol.y[:, -1].item()

        # humidity ode solver
        rh_out = self.outside_rh_humid[self.time]
        w_abs_out = self.map_rel_to_abs_humid(self.inside_temp, rh_out)
        humid_args = (E, w_abs_out)
        humid_sol = solve_ivp(self.water_balance, t_span, [self.inside_abs_humid], args=humid_args)
        new_humid = humid_sol.y[:, -1].item()

        return new_temp, new_humid

    def report(self) -> pd.DataFrame:
        # TODO: FIGURE OUT HOW INCLUDE ACTIONS IN REPORT
        """probably use a pandas dataframe for display"""
        d = {"Outside Temp": self.outside_temp[:-1],
             "Initial Temp": self.start_temp_hist,
             "Final Temp": self.final_temp_history,
             "Initial Inside RH": self.initial_rel_humid_history,
             "Final Inside RH": self.final_rel_humid_history,
             }

        df = pd.DataFrame(d)
        return df


if __name__ == '__main__':
    env = GreenhouseBaseEnv()
