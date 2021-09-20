import numpy as np
import matplotlib.pyplot as plt
#
# daily_temp = []
# temp = 22
# sign = -1
# for i in range(24):
#     if i % 4 == 0:
#         sign *= -1
#     if i % 2 == 0:
#         sign *=
#     temp += 1.5 * sign
#     daily_temp.append(temp)
# print(daily_temp)
swing = np.arange(6)
swing = 1.5 * swing
base_line = 0
temps = np.array((swing + base_line, base_line + swing[::-1], np.full(6, base_line), np.full(6, base_line))).flatten()
x = np.arange(24)
y = temps

plt.plot(x,y)
plt.show()