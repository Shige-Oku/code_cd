#coding: UTF-8

import matplotlib.pyplot as plt
import numpy as np
import math

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def d_sigmoid(a):
    return sigmoid(a) * (1 - sigmoid(a))

e = math.e
dx = 0.1
x = np.arange(-8, 8, dx)

# シグモイド関数： y = 1 / (1 + e^(-x))
y_sig = sigmoid(x)
# シグモイド関数の傾き（微分）
# y_dsig = (sigmoid(x+dx) - sigmoid(x)) / dx
y_dsig = sigmoid(x) * (1 - sigmoid(x))
dmax = max(y_dsig)

# 1 / 1 + e^-x
plt.title("sigmoid function")
plt.grid()
plt.plot(x, y_sig, label="sigmoid")
plt.plot(x, y_dsig, label="dy_sigmoid")
plt.legend()
plt.show()
