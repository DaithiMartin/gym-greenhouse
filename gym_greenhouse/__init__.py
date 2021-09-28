from gym.envs.registration import register

register(
    id='greenhouse-discrete-v0',
    entry_point='gym_greenhouse.envs:GreenhouseDiscreteEnv',
)

register(
    id='greenhouse-continuous-v0',
    entry_point='gym_greenhouse.envs:GreenhouseContinuousEnv'
)
