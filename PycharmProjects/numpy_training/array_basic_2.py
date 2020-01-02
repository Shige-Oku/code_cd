# coding: UTF-8

import numpy as np

a = np.arange(42).reshape(6, 7)
print(np.shape(a))
print(np.size(a))

(row, col) = np.shape(a)
print(row)
print(col)

b = np.zeros(10)
print(b)

c = np.ones(10)
print(c)

d = np.random.rand(10)
print(d)

e = np.random.permutation(range(10))
print(e)

print(a.shape)
print(a.size)
