import matplotlib.pyplot as plt
import numpy as np
import math

def sigmoid(a):
    return 1 / (1 + e**-x)

e = math.e
dx = 0.1
x = np.arange(-8, 8, dx)

y_sig = sigmoid(x)
print(x)

# 1 / 1 + e^-x
plt.plot(x, y_sig)
plt.show()
