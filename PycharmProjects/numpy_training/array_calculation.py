# coding: UTF-8

import numpy as np

a = np.arange(6).reshape(2, 3)
print(a)

b = a + 1
print(b)

c = a * 2
print(c)

d = a**2
print(d)

e = np.sum(a)
print(e)

f = np.mean(a)
print(f)

g = np.arange(1, 7).reshape(2, 3)
print(g)

h = a + g
print(h)

i = a * g
print(i)
