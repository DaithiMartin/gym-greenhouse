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
ACTION_MAX: float = 10  # watts
ACTION_MIN: float = -10  # watts

# Greenhouse simulation
SPECIFIC_HEAT: float = 1005.0  # J * kg^-1 K^-1, specific heat of "ambient" air
AIR_DENSITY: float = 1.225  # kg / m^3

SIM_TIME = 3 * 24   # hours

# OUTSIDE_RH = 60
WEATHER_DATA_PATH = "../../weather_data/sample_data.csv"
RANDOM_WEATHER_SAMPLE = False
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

    Underlying Assumptions
    - no condensation forms on the inside of the glazing surfaces
    - no evapotranspiration from ground
    - only source of humidity inside the GH is from evapotranspiration
    - only loss in humidity is from ventilation

    """
    metadata = {'render.modes': ['human']}

    def __init__(self):

        super(GreenhouseBaseEnv, self).__init__()

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

        self.outside_rh_humid = None
        self.outside_temp = None
        self.solar_radiation: ArrayLike = self.set_radiative_heat()
        self.generate_outside_environment()

        # self.outside_temp: ArrayLike = self.set_outside_temp()  # set outside temp for episode

        self.inside_temp: int = self.outside_temp[0]      # starting inside temp
        self.ideal_temp: int = 27       # ideal inside temp
        self.temp_tolerance: int = 2    # +/- tolerance for ideal temp
        # self.outside_rh_humid: ArrayLike = self.set_outside_rh()    # set relative humidity for episode

        self.outside_ah_humid: ArrayLike = self.map_rel_to_abs_humid(self.outside_temp, self.outside_rh_humid)
        self.inside_abs_humid: float = self.map_rel_to_abs_humid(self.inside_temp, self.outside_rh_humid[0])   # initial inside humid

        self.ideal_rel_humid: float = self.outside_rh_humid[0]   # ideal relative humidity in percentage form
        self.humid_tolerance: float = 5  # +/- tolerance for ideal humidity in percentage form

        # histories
        self.reward_history: List[float] = []  # reward history for episode
        self.action_history: List[T] = []  # action history for episode
        # self.temp_change_history: list = []  # env temp change history for episode
        # self.rad_temp_change_history: list = []  # radiative component of env temp change history for episode

        self.start_temp_hist: List[float] = []
        self.final_temp_history: List[float] = []  # internal temp history for episode
        self.initial_rel_humid_history: List[float] = []
        self.final_rel_humid_history: List[float] = []
        self.abs_humidity_history = []

        # ODE components
        self.ode_heat_radiation_gain = []
        self.ode_heat_heater_gain = []
        self.ode_heat_sensible_loss = []
        self.ode_heat_latent_loss = []
        self.ode_heat_conductive_loss = []
        self.ode_humid_vent_loss = []
        self.ode_humid_vent_gain = []
        self.ode_humid_evap_gain = []
        self.ode_humid_total = []

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

    def step(self,
             action):
        """
        Take a step in the environment.
        """

        state = self.update_state(action)
        reward = self.get_reward(action)
        done = False if self.time < SIM_TIME else True
        info = None

        return state, reward, done, info

    def reset(self):
        """Re-initialize Environment and return starting state"""

        self.__init__()
        state = self.get_state()

        return state

    def render(self,
               mode: str = 'human',
               report: bool = False):

        """Display episode visualization."""

        # internal vs external temperature
        x = np.arange(SIM_TIME)
        temp_external = self.outside_temp[:-1]  # exclude last time because that's hour 25
        temp_internal = self.final_temp_history
        temp_ideal = np.full(SIM_TIME, self.ideal_temp)
        plt.plot(x, temp_external, 'v-', label="External")
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
        humid_ideal = np.full(SIM_TIME, self.ideal_rel_humid)
        plt.plot(x, humid_external, 'v-', label="External")
        plt.plot(x, humid_internal, 'o-', label="Internal")
        plt.fill_between(x, humid_ideal + self.humid_tolerance, humid_ideal - self.humid_tolerance,
                         label="Ideal", alpha=0.3, color='y')
        plt.title("Humidity")
        plt.xlabel("Time")
        plt.ylabel("Relative Humidity")
        plt.legend()
        plt.show()

        # abs_humid = self.abs_humidity_history
        # plt.plot(x, abs_humid, label="Internal")
        # plt.title("Absolute Humidity")
        # plt.ylabel("Absolute Humidity (g water / kg air)")
        # plt.xlabel("Time")
        # plt.legend()
        # plt.show()

        if report:
            self.episode_report = self.report()

        return None

    def close(self):
        pass

    def generate_outside_environment(self):
        data = pd.read_csv(WEATHER_DATA_PATH, index_col=0)
        data.index = pd.to_datetime(data.index)

        sample_range = pd.date_range(data.index[0], data.index[-72], freq="D")

        if RANDOM_WEATHER_SAMPLE:
            index = np.random.randint(0, len(sample_range) - 3)
        else:
            index = 0

        sample = data.loc[sample_range[index]: sample_range[index + 3]]
        self.outside_temp = sample["air_temp_set_1"].array
        self.outside_rh_humid = sample["relative_humidity_set_1"].array
        return None

    @staticmethod
    def set_outside_temp():

        base = np.arange(6)
        base = 1.5 * base
        temps = np.array((base + 22, 22 + base[::-1], 22 - base, 22 - base[::-1])).flatten()
        temps = np.concatenate((temps, temps, temps))
        temps = np.concatenate((temps, np.array([22])))

        return temps

    @staticmethod
    def set_outside_rh():
        """RH in percentage form, eg: 20.0% not 0.20"""

        # rel_humid = np.full(SIM_TIME + 1, OUTSIDE_RH)
        return None

    @staticmethod
    def set_outside_ah():
        abs_humid = np.full(SIM_TIME + 1, 5)
        return abs_humid

    def get_reward(self, action):

        inside_temp = self.inside_temp
        ideal_temp = self.ideal_temp
        temp_tolerance = self.temp_tolerance

        if ideal_temp + temp_tolerance >= inside_temp >= ideal_temp - temp_tolerance:
            temp_reward = 1000
        else:
            temp_reward = -((inside_temp - ideal_temp) ** 2)

        inside_rh = self.map_abs_to_rel_humid(inside_temp, self.inside_abs_humid)
        ideal_humid = self.ideal_rel_humid
        humid_tolerance = self.humid_tolerance
        if ideal_humid + humid_tolerance >= inside_rh >= ideal_humid - humid_tolerance:
            humid_reward = 100
        else:
            humid_reward = -(np.abs(inside_rh - ideal_humid))

        # TODO: THIS COULD BE WEIGHTED
        reward = temp_reward + humid_reward

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
        new_temp, new_rel_humid = self.get_temp_humid(action)

        # update instance state
        self.inside_temp = new_temp
        self.inside_abs_humid = self.map_rel_to_abs_humid(new_temp, new_rel_humid)

        # update histories
        self.final_temp_history.append(self.inside_temp)
        self.abs_humidity_history.append(self.inside_abs_humid)
        # self.final_rel_humid_history.append(self.map_abs_to_rel_humid(self.inside_temp, self.inside_abs_humid))
        self.final_rel_humid_history.append(new_rel_humid)
        self.action_history.append((self.action_map(action[0]), self.action_map(action[1])))

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

    @staticmethod
    def set_radiative_heat():

        # peak Q_GRout
        max_Q_GRout = 1025  # W m^-2
        day_length = 12
        day = np.sin(np.linspace(0, np.pi, day_length)) * max_Q_GRout
        night = np.zeros(6)

        morning = np.zeros(6)

        full_day = np.concatenate((morning, day, night))
        Q_GRout = np.concatenate((full_day, full_day, full_day))
        Q_GRout = np.concatenate((Q_GRout, np.array([0])))

        return Q_GRout

    @staticmethod
    def map_rel_to_abs_humid(temp, rel_humid):
        """Maps relative humidity -> absolute humidity
        RH in percentage form eg: 20% not 0.20
        source: https://carnotcycle.wordpress.com/2012/08/04/how-to-convert-relative-humidity-to-absolute-humidity/"""
        abs_humid = (6.112 * np.exp((17.67 * temp) / (temp + 243.5)) * rel_humid * 2.1674)
        abs_humid = abs_humid / (273.15 + temp)

        return abs_humid

    @staticmethod
    def map_abs_to_rel_humid(temp, abs_humid):
        """Masp absolute humidity -> relative humidity(non-decimal
        RH in percentage form eg: 20% not 0.20
        source: https://carnotcycle.wordpress.com/2012/08/04/how-to-convert-relative-humidity-to-absolute-humidity/"""
        rel_humid = abs_humid * (273.15 + temp)
        rel_humid = rel_humid / (6.112 * np.exp((17.67 * temp) / (temp + 243.5)) * 2.1674)

        return rel_humid

    def get_q_heater(self,
                     action: tuple):

        ground_surface = self.width * self.length
        heater_capacity = self.action_map(action)
        Q_heater = self.num_heaters * heater_capacity / ground_surface

        return Q_heater

    @staticmethod
    def get_latent_sensible_loss(evaporation,
                                 inside_temp,
                                 outside_temp):

        L = 2.5e6  # latent heat of vaporization (J kg^-1)
        qv = 0.003  # ventilation rate (m^3 m^-2 s^-1)
        Cp = 1010  # specific heat of moist air (J kg^-1 K^-1)
        rho = 1.2  # specific mass of air  (kg dry air m^-3)

        latent = L * evaporation
        sensible = qv * Cp * rho * (inside_temp - outside_temp)

        return latent, sensible

    @ staticmethod
    def get_conductive_heat_loss(inside_temp,
                                 outside_temp):

        w = 2.1  # ratio of glazing to floor area
        k = 6.2  # heat transfer coefficient (W m^-2 C^-1)

        conductive_loss = k * w * (inside_temp - outside_temp)

        return conductive_loss

    @staticmethod
    def get_humid_vent_bal(inside_humidity,
                           outside_humidity):

        qv = 0.003  # ventilation rate (m^3 m^-2 s^-1)
        rho = 1.2  # specific mass of air  (kg dry air m^-3)

        vent_loss = qv * rho * inside_humidity
        vent_gain = qv * rho * outside_humidity

        return vent_loss, vent_gain

    @staticmethod
    def get_new_abs_out(w_in,
                        evapo_trans):
        qv = 0.003  # ventilation rate (m^3 m^-2 s^-1)
        rho = 1.2  # specific mass of air  (kg dry air m^-3)

        d_abs_out = w_in * qv * rho - evapo_trans
        d_abs_out = d_abs_out / (qv * rho)

        return d_abs_out

    def energy_bal(self, t, T_in, *args):
        """RHS of temperature energy balance equation"""

        # constant parameters
        Cp = 1010  # specific heat of moist air (J kg^-1 K^-1)
        rho = 1.2  # specific mass of air  (kg dry air m^-3)
        H = self.height  # Note that the paper calls for avg height but does not describe how that is calculated(m)
        L = 2.5e6  # latent heat of vaporization (J kg^-1)
        qv = 0.003  # ventilation rate (m^3 m^-2 s^-1)
        # TODO: DETERMINE HOW TO MAKE THIS DYNAMIC WITH DIFFERENT GH ARCHITECTURE
        w = 2.1  # ratio of glazing to floor area
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

        **note 1 W := J s^-1
        """

        # Q_heater
        # Q_heater = self.get_q_heater(action)
        Q_heater = 0

        # Q_GRin
        tau_c = 0.9  # solar radiation transmittance of glazing material (dimensionless)
        rho_g = 0.5  # reflectance of solar radiation on ground (dimensionless)
        Q_GRout = self.solar_radiation[self.time]  # global outside radiation (W m^-2)
        Q_GRin = tau_c * (1 - rho_g) * Q_GRout

        # latent heat
        # energy consumed in the phase transition of water, in this simulation: evapotranspiration
        E = (3e-4 * tau_c * Q_GRin + 0.0021) / 15 / 60  # evapotranspiration rate (kg m^-2 15min^-1) /15min /60 sec
        E = 0  # no plants

        # ODE parameters
        t_span = (0, 60 * 60)   # ode time period, 60 sec min^-1 * 60 min hr^-1
        T_out = self.outside_temp[self.time]
        T_in = self.inside_temp
        # T_in += self.action_map(action)

        # energy ode solver
        energy_args = (Q_GRin, Q_heater, E, T_out)
        energy_sol = solve_ivp(self.energy_bal, t_span, [T_in], args=energy_args)
        new_temp = energy_sol.y[:, -1].item()
        new_temp += self.action_map(action[0])

        # log heat components
        self.ode_heat_heater_gain.append(Q_heater)
        self.ode_heat_radiation_gain.append(Q_GRin)
        latent_loss, sensible_loss = self.get_latent_sensible_loss(E, new_temp, T_out)
        self.ode_heat_latent_loss.append(-latent_loss)
        self.ode_heat_sensible_loss.append(-sensible_loss)
        conductive_loss = self.get_conductive_heat_loss(new_temp, T_out)
        self.ode_heat_conductive_loss.append(-conductive_loss)

        # humidity ode solver
        # rh_out = self.outside_rh_humid[self.time]
        # w_abs_out = self.map_rel_to_abs_humid(self.inside_temp, rh_out)
        W_in = self.inside_abs_humid
        w_abs_out = self.outside_ah_humid[self.time]
        humid_args = (E, w_abs_out)
        humid_sol = solve_ivp(self.water_balance, t_span, [W_in], args=humid_args)
        new_in_humid = humid_sol.y[:, -1].item()   # absolute humidity
        new_rel_in_humid = self.map_abs_to_rel_humid(new_temp, new_in_humid)
        new_rel_in_humid += self.action_map(action[1])
        delta_humid_out = -self.get_new_abs_out(new_in_humid, E) #(new_in_humid - W_in)   # calc gain in absolute humidity to climate
        # self.outside_ah_humid[self.time + 1] = self.outside_ah_humid[self.time + 1] + delta_humid_out   # update outside AH

        # log humidity components
        vent_loss, vent_gain = self.get_humid_vent_bal(new_in_humid, w_abs_out + delta_humid_out)
        self.ode_humid_vent_loss.append(-vent_loss)
        self.ode_humid_vent_gain.append(vent_gain)
        self.ode_humid_evap_gain.append(E)
        self.ode_humid_total.append(E + vent_gain - vent_loss)

        # checks
        check_energy = Q_GRin + Q_heater - latent_loss - sensible_loss - conductive_loss
        # check_humid = vent_loss - vent_gain - E

        return new_temp, new_rel_in_humid

    def report(self) -> pd.DataFrame:
        # TODO: FIGURE OUT HOW INCLUDE ACTIONS IN REPORT
        """probably use a pandas dataframe for display"""
        d = {"Outside Temp": self.outside_temp[:-1],
             "Initial Temp": self.start_temp_hist,
             "Final Temp": self.final_temp_history,
             "Initial Inside RH": self.initial_rel_humid_history,
             "Final Inside RH": self.final_rel_humid_history,
             "Global Radiation Gain": self.ode_heat_radiation_gain,
             "Heater Gain": self.ode_heat_heater_gain,
             "Sensible Heat Loss": self.ode_heat_sensible_loss,
             "Latent Heat Loss": self.ode_heat_latent_loss,
             "Conductive Heat Loss": self.ode_heat_conductive_loss,
             "Humidity Vent Loss": self.ode_humid_vent_loss,
             "Humidity Vent Gain": self.ode_humid_vent_gain,
             "Humidity Evapotranspiration Gain": self.ode_humid_evap_gain,
             "Humidity Total": self.ode_humid_total,
             }

        # heat component plots
        x = np.arange(SIM_TIME)
        plt.plot(x, d["Global Radiation Gain"], label="Global Radiation Gain")
        plt.plot(x, d["Heater Gain"], label="Heater Gain")
        plt.plot(x, d["Sensible Heat Loss"], label="Sensible Heat Loss")
        plt.plot(x, d["Latent Heat Loss"], label="Latent Heat Loss")
        plt.plot(x, d["Conductive Heat Loss"], label="Conductive Heat Loss")
        plt.title("Heat ODE Components")
        plt.xlabel("Time")
        plt.ylabel("Heat (W m^-2)")
        plt.legend()
        plt.show()

        # humidity components
        # plt.plot(x, d["Humidity Vent Loss"], label="Humidity Vent Loss")
        # plt.plot(x, d["Humidity Vent Gain"], label="Humidity Vent Gain")
        # plt.plot(x, d["Humidity Evapotranspiration Gain"], label="Humidity Evapotranspiration Gain")
        # plt.plot(x, d["Humidity Total"], label="Humidity Total")
        # plt.title("Humidity ODE Components")
        # plt.xlabel("Time")
        # plt.ylabel("Absolute Humidity (g water kg^-1 dry air")
        # plt.legend()
        # plt.show()

        df = pd.DataFrame(d)
        return df


if __name__ == '__main__':
    env = GreenhouseBaseEnv()
