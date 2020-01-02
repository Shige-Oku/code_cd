import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10, 0.1)

# y = x ^ 2 + 10x + 10
y_2 = x**2 - 10*x + 10

plt.plot(x, y_2)
# plt.show()

x_2 = np.arange(-5, 15, 0.1)
# y = x^3 -10x^2 -10x +10
y_3 = x_2**3 - 10*x_2**2 + 10*x_2 + 10

plt.plot(x_2, y_3)
plt.show()


