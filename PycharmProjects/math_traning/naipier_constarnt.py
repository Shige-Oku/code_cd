import matplotlib.pyplot as plt
import numpy as np
import math

e = math.e
print(e)

dx = 0.1
x = np.arange(-5, 5, dx)

y_2 = 2**x
y_e = e**x
y_3 = 3**x

# y = (e^(x+dx) -e^x) / dx
y_de = (e**(x+dx) - e**x) / dx

# plt.plot(x, y_2)
plt.plot(x, y_e, label="x**2")
# plt.plot(x, y_3)
plt.plot(x, y_de, label="e")

dx2 = 0.00001
x_e2 = np.arange(-5, 5, dx2)
y_de2 = (e**(x_e2+dx2) - e**x_e2) / dx2

plt.plot(x_e2, y_de2, label="e'")
plt.legend()
plt.show()
