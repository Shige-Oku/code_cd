# coding: UTF-8

import chainer
from chainer import Variable, Chain, optimizers, serializers, datasets
import chainer.links as L
import chainer.functions as F

from chainer.datasets import tuple_dataset
from chainer import training, iterators
from chainer.training import extensions

import numpy as np
import matplotlib.pyplot as plt

# -- MNISTデータの読み込み --
# -- 1 * 28 * 28 の３次元の画像として読み込む --
mnist_data = datasets.get_mnist(ndim=3)
train_data = mnist_data[0]
test_data = mnist_data[1]

print("Train:", len(train_data))
print("Test:", len(test_data))

# -- MNIST画像の表示 --

index = 2
plt.imshow(train_data[index][0].reshape(28, 28), cmap='gray')
plt.title(train_data[index][1])
plt.show()
