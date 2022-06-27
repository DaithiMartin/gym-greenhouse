import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Union, Callable, Any, List, TypeVar, Generic
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp

# Hyper parameters
# -------------------------------------------------------------------------------#
# initial conditions
INSIDE_TEMP = 22
INSIDE_REL_HUMIDITY = 30

# environment bounds
TEMP_MAX = 100
TEMP_MIN = -50
HUMID_MAX = 100
HUMID_MIN = 0
RADIATION_MAX = 1200
RADIATION_MIN = 0

# ideal environment parameters
IDEAL_TEMP = 22
TEMP_TOLERANCE = 3
IDEAL_REL_HUMID = 50
HUMID_TOLERANCE = 10

# action parameters
TEMP_ACTION_MAX = 10
TEMP_ACTION_MIN = -10
HUMID_ACTION_MAX = 10
HUMID_ACTION_MIN = -10

# reward parameters
TEMP_REWARD_WEIGHT = 1
HUMID_REWARD_WEIGHT = 0

# Greenhouse simulation
HEIGHT = 6.3
LENGTH = 30
WIDTH = 10
SPECIFIC_HEAT: float = 1005.0  # J * kg^-1 K^-1, specific heat of "ambient" air
AIR_DENSITY: float = 1.225  # kg / m^3
SIM_TIME = 3 * 24  # hours

# weather data
WEATHER_DATA_PATH = "./weather_data/sample_data.csv"
RANDOM_WEATHER_SAMPLE = False


# -------------------------------------------------------------------------------#


class NewGreenhouseBase(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(NewGreenhouseBase, self).__init__()

        self.temp_action_max = TEMP_ACTION_MAX
        self.temp_action_min = TEMP_ACTION_MIN
        self.humid_action_max = HUMID_ACTION_MAX
        self.humid_action_min = HUMID_ACTION_MIN
        self.action_map = self.get_action_map()  # maps agent action to environmental response

        # gym spaces
        self.action_space = self.get_action_space()
        self.observation_space = gym.spaces.Box(low=np.zeros(7, dtype=np.float32),
                                                high=np.ones(7, dtype=np.float32)
                                                )

        # environment bounds
        self.temp_max = TEMP_MAX
        self.temp_min = TEMP_MIN
        self.humid_max = HUMID_MAX
        self.humid_min = HUMID_MIN
        self.radiation_max = RADIATION_MAX
        self.radiation_min = RADIATION_MIN

        # ideal conditions
        self.ideal_temp = IDEAL_TEMP
        self.temp_tolerance = TEMP_TOLERANCE
        self.ideal_rel_humid = IDEAL_REL_HUMID
        self.humid_tolerance = HUMID_TOLERANCE

        # reward parameters
        self.temp_reward_weight = TEMP_REWARD_WEIGHT
        self.humid_reward_weight = HUMID_REWARD_WEIGHT

        # greenhouse physical parameters
        self.height = HEIGHT
        self.length = LENGTH
        self.width = WIDTH

        # instance variables
        self.time = 0
        self.inside_temp = INSIDE_TEMP
        self.inside_rel_humid = INSIDE_REL_HUMIDITY

        self.outside_temp, self.outside_rel_humid, self.solar_radiation = self.generate_outside_environment()

        # histories
        self.init_temp_hist = []
        self.init_rel_humid_hist = []
        self.final_temp_hist = []
        self.final_rel_humid_hist = []
        self.reward_hist = []

        self.episode_report = None

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

        # report
        self.episode_report = None

    def step(self, action):
        state = self.update_state(action)
        reward = self.get_reward()
        done = False if self.time < SIM_TIME else True
        info = {}

        return state, reward, done, info

    def reset(self):
        self.__init__()
        state = self.get_state()
        return state

    def render(self, mode='human', report=False):

        # internal vs external temperature
        x = np.arange(SIM_TIME)
        temp_external = self.outside_temp[:-1]  # exclude last time because that's hour 25
        temp_internal = self.final_temp_hist
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
        humid_external = self.outside_rel_humid[:-1]
        humid_internal = self.final_rel_humid_hist
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

        if report:
            self.episode_report = self.report()

        return None

    def get_action_space(self):
        raise NotImplementedError("Action space not defined.")
        pass

    def get_action_map(self):
        raise NotImplementedError("action map not defined")
        pass

    def update_state(self, action):
        # log initial temperature
        self.init_temp_hist.append(self.inside_temp)
        self.init_rel_humid_hist.append(self.inside_rel_humid)

        # agent takes action and environment reacts
        new_temp, new_rel_humid = self.get_new_temp_humid(action)

        # update instance state variables
        self.inside_temp = np.clip(new_temp, self.temp_min, self.temp_max)
        self.inside_rel_humid = np.clip(new_rel_humid, self.humid_min, self.humid_max)

        # update finial histories
        self.final_temp_hist.append(self.inside_temp)
        self.final_rel_humid_hist.append(self.inside_rel_humid)

        # increment time
        self.time += 1

        # define state at new time step
        state = self.get_state()

        return state

    def get_state(self):
        """
        Gets current state of environment.
        This is its own function because both update_state(action) and reset() need to return the full state.
        All components of state are normalized to [0,1] for learning stability
        """

        state = [self.normalize_temp(self.ideal_temp),
                 self.normalize_temp(self.inside_temp),
                 self.normalize_temp(self.outside_temp[self.time]),
                 self.normalize_humidity(self.ideal_rel_humid),
                 self.normalize_humidity(self.inside_rel_humid),
                 self.normalize_humidity(self.outside_rel_humid[self.time]),
                 self.normalize_radiation(self.solar_radiation[self.time])
                 ]
        return np.array(state)

    def get_reward(self):

        # temp reward component
        inside_temp = self.inside_temp
        ideal_temp = self.ideal_temp
        temp_tolerance = self.temp_tolerance

        if ideal_temp + temp_tolerance >= inside_temp >= ideal_temp - temp_tolerance:
            temp_reward = 1
        else:
            # decrease reward by 1/2 for every 2 degrees away from ideal range
            if inside_temp > (ideal_temp + temp_tolerance):
                dif = inside_temp - ideal_temp + temp_tolerance
            else:
                dif = ideal_temp + temp_tolerance - inside_temp
            steps = np.ceil(dif / 2)
            temp_reward = 1.0 * 0.5 ** steps

        # humidity reward component
        inside_rh = self.inside_rel_humid
        ideal_humid = self.ideal_rel_humid
        humid_tolerance = self.humid_tolerance
        if ideal_humid + humid_tolerance >= inside_rh >= ideal_humid - humid_tolerance:
            humid_reward = 1
        else:
            # decrease reward by 1/2 for every 2 degrees away from ideal range
            if inside_rh > (ideal_humid + humid_tolerance):
                dif = inside_rh - ideal_humid + humid_tolerance
            else:
                dif = ideal_humid + humid_tolerance - inside_rh

            steps = np.ceil(dif / 2)
            humid_reward = 1.0 * 0.5 ** steps

        # weighted reward in domain [0,1]
        reward = self.temp_reward_weight * temp_reward + self.humid_reward_weight * humid_reward

        # update history
        self.reward_hist.append(reward)

        return reward

    def get_new_temp_humid(self, action):
        """
        Energy and Mass Balance for internal heat and humidity.

        Heat, all units: W m^-2
        Q_GRin + Q_heater = Q_IV + Q_glazing

        Water, absolute humidity in kg water kg^-1 dry air
        W_in * q_v * rho = W_out * q_v * rho + E

        **note 1 W := J s^-1

        action: tuple(temp_action, humid_action)
        """
        temp_action, humid_action = self.action_map(action)

        # Q_heater
        # Q_heater = self.get_q_heater(action)
        # FIXME: ALLOW AGENT TO CONTROL Q_heater
        Q_heater = 0

        # Q_GRin
        tau_c = 0.9  # solar radiation transmittance of glazing material (dimensionless)
        rho_g = 0.5  # reflectance of solar radiation on ground (dimensionless)
        Q_GRout = self.solar_radiation[self.time]  # global outside radiation (W m^-2)
        Q_GRin = tau_c * (1 - rho_g) * Q_GRout

        # latent heat
        # energy consumed in the phase transition of water, in this simulation: evapotranspiration
        E = (3e-4 * tau_c * Q_GRin + 0.0021) / 15 / 60  # evapotranspiration rate (kg m^-2 15min^-1) /15min /60 sec
        # E = 0  # no plants

        # ODE parameters
        t_span = (0, 60 * 60)  # ode time period, 60 sec min^-1 * 60 min hr^-1
        T_out = self.outside_temp[self.time]
        T_in = self.inside_temp

        # energy ode solver
        energy_args = (Q_GRin, Q_heater, E, T_out)
        energy_sol = solve_ivp(self.energy_bal, t_span, [T_in], args=energy_args)
        new_temp = energy_sol.y[:, -1].item()
        # agent action
        new_temp += temp_action

        # log heat components
        self.ode_heat_heater_gain.append(Q_heater)
        self.ode_heat_radiation_gain.append(Q_GRin)
        latent_loss, sensible_loss = self.get_latent_sensible_loss(E, new_temp, T_out)
        self.ode_heat_latent_loss.append(-latent_loss)
        self.ode_heat_sensible_loss.append(-sensible_loss)
        conductive_loss = self.get_conductive_heat_loss(new_temp, T_out)
        self.ode_heat_conductive_loss.append(-conductive_loss)

        # humidity ode solver
        W_in = self.map_rel_to_abs_humid(self.inside_temp, self.inside_rel_humid)
        w_abs_out = self.map_rel_to_abs_humid(self.outside_temp[self.time], self.outside_rel_humid[self.time])
        humid_args = (E, w_abs_out)
        humid_sol = solve_ivp(self.water_balance, t_span, [W_in], args=humid_args)
        new_in_humid = humid_sol.y[:, -1].item()  # absolute humidity
        new_rel_in_humid = self.map_abs_to_rel_humid(new_temp, new_in_humid)
        # agent action
        new_rel_in_humid += humid_action

        # log humidity components
        vent_loss, vent_gain = self.get_humid_vent_bal(new_in_humid, w_abs_out)
        self.ode_humid_vent_loss.append(-vent_loss)
        self.ode_humid_vent_gain.append(vent_gain)
        self.ode_humid_evap_gain.append(E)
        self.ode_humid_total.append(E + vent_gain - vent_loss)

        # checks
        check_energy = Q_GRin + Q_heater - latent_loss - sensible_loss - conductive_loss
        check_humid = vent_loss - vent_gain

        return new_temp, new_rel_in_humid

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

    @staticmethod
    def get_latent_sensible_loss(evaporation, inside_temp, outside_temp):

        L = 2.5e6  # latent heat of vaporization (J kg^-1)
        qv = 0.003  # ventilation rate (m^3 m^-2 s^-1)
        Cp = 1010  # specific heat of moist air (J kg^-1 K^-1)
        rho = 1.2  # specific mass of air  (kg dry air m^-3)

        latent = L * evaporation
        sensible = qv * Cp * rho * (inside_temp - outside_temp)

        return latent, sensible

    @staticmethod
    def get_conductive_heat_loss(inside_temp, outside_temp):

        w = 2.1  # ratio of glazing to floor area
        k = 6.2  # heat transfer coefficient (W m^-2 C^-1)

        conductive_loss = k * w * (inside_temp - outside_temp)

        return conductive_loss

    @staticmethod
    def get_humid_vent_bal(inside_humidity, outside_humidity):

        qv = 0.003  # ventilation rate (m^3 m^-2 s^-1)
        rho = 1.2  # specific mass of air  (kg dry air m^-3)

        vent_loss = qv * rho * inside_humidity
        vent_gain = qv * rho * outside_humidity

        return vent_loss, vent_gain

    @staticmethod
    def map_abs_to_rel_humid(temp, abs_humid):
        """Masp absolute humidity -> relative humidity(non-decimal
        RH in percentage form eg: 20% not 0.20
        source: https://carnotcycle.wordpress.com/2012/08/04/how-to-convert-relative-humidity-to-absolute-humidity/"""
        rel_humid = abs_humid * (273.15 + temp)
        rel_humid = rel_humid / (6.112 * np.exp((17.67 * temp) / (temp + 243.5)) * 2.1674)

        return rel_humid

    @staticmethod
    def map_rel_to_abs_humid(temp, rel_humid):
        """Maps relative humidity -> absolute humidity
        RH in percentage form eg: 20% not 0.20
        source: https://carnotcycle.wordpress.com/2012/08/04/how-to-convert-relative-humidity-to-absolute-humidity/"""
        abs_humid = (6.112 * np.exp((17.67 * temp) / (temp + 243.5)) * rel_humid * 2.1674)
        abs_humid = abs_humid / (273.15 + temp)

        return abs_humid

    @staticmethod
    def set_radiative_heat():
        # FIXME: TO BE DEPRECIATED ONCE REAL WORLD DATA IS AVAILABLE
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

    def normalize_temp(self, temp):
        """Normalizes temperature to domain [0,1] for agent learning stability"""
        norm_temp = temp - self.temp_min
        norm_temp /= (self.temp_max - self.temp_min)
        return norm_temp

    def normalize_radiation(self, radiation):
        """Normalizes radiation to domain [0,1] for agent learning stability"""
        norm_radiation = radiation - self.radiation_min
        norm_radiation /= (self.radiation_max - self.radiation_min)
        return norm_radiation

    def normalize_humidity(self, humidity):
        """Normalizes humidity to domain [0,1] for agent learning stability"""
        norm_humid = humidity - self.humid_min
        norm_humid /= (self.humid_max - self.humid_min)
        return norm_humid

    def generate_outside_environment(self):
        data = pd.read_csv(WEATHER_DATA_PATH, index_col=0)
        data.index = pd.to_datetime(data.index)

        sample_range = pd.date_range(data.index[0], data.index[-72], freq="D")

        if RANDOM_WEATHER_SAMPLE:
            index = np.random.randint(0, len(sample_range) - 3)
        else:
            index = 0

        sample = data.loc[sample_range[index]: sample_range[index + 3]]
        outside_temp = sample["air_temp_set_1"].array
        outside_rel_humid = sample["relative_humidity_set_1"].array
        # FIXME: PUT REAL RADIATIVE HEAT IN DATAFRAME
        outside_solar_radiation = self.set_radiative_heat()

        return outside_temp, outside_rel_humid, outside_solar_radiation

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


class NewGreenhouseContinuousEnv(NewGreenhouseBase):
    def __init__(self):
        super(NewGreenhouseContinuousEnv, self).__init__()
        self.action_space = self.get_action_space()
        self.action_map = self.get_action_map()

    def get_action_space(self):
        action_space = gym.spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
        )

        return action_space

    def get_action_map(self):
        """returns function handle for mapping [0,1] domain from agent to environmental output"""
        def action_map(action):
            temp_action = (action[0] * (self.temp_action_max - self.temp_action_min)) + self.temp_action_min
            humid_action = (action[1] * (self.humid_action_max - self.humid_action_min)) + self.humid_action_min

            return temp_action, humid_action

        return action_map

if __name__ == '__main__':
    env = NewGreenhouseContinuousEnv()
    observation = env.reset()
    for t in range(80):
        action = np.array([0.5, 0.5])
        # action_check = env.action_map(action)
        observation, reward, done, info = env.step(action)
        if done:
            print(f"Episode finished after {t + 1} time-steps")
            break
    env.render()

print("complete")
