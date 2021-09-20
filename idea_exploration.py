import numpy as np
import matplotlib.pyplot as plt

# swing = np.arange(6)
# swing = 1.5 * swing
# base_line = 0
# temps = np.array((swing + base_line, base_line + swing[::-1], np.full(6, base_line), np.full(6, base_line))).flatten()
# x = np.arange(24)
# y = temps
#
# plt.plot(x,y)
# plt.show()

action_min = -10
action_max = 10
action_range = range(action_min, action_max + 1)
num_actions = 21
index_range = range(num_actions)
action_map = {}
for index, action in zip(index_range, action_range):
    action_map[index] = action
print(action_map)