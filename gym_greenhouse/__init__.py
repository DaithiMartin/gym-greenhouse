from gym.envs.registration import register

register(
    id='greenhouse-v0',
    entry_point='gym_greenhouse.envs:GreenhouseEnv',
)

register(
    id='greenhouse-continuous-v0',
    entry_point='gym_greenhouse.envs:ContinuousGreenhouseEnv'
)
