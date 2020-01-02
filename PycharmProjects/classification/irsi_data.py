# coding: UTF-8

import chainer
from chainer import Variable, Chain, optimizers
import chainer.links as L
import chainer.functions as F

import numpy as np
from sklearn import datasets

# -- Iris データ読み込み --
iris_data = datasets.load_iris()
# print(iris_data)

x = iris_data.data.astype(np.float32)
t = iris_data.target
n = t.size

print(x)
print(t)
print(n)
