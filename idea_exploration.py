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
base = np.arange(6)
base = 1.5 * base
temps = np.array((base + 22, 22 + base[::-1], 22 - base, 22 - base[::-1])).flatten()
x = np.arange(24)
y = temps

plt.plot(x,y)
plt.show()