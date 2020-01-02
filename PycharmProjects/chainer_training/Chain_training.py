# coding: UTF-8

import chainer
import numpy as np
from chainer import Variable, Chain
import chainer.links as L

# -- 関数で記述 --
l1 = L.Linear(4, 3)
l2 = L.Linear(3, 2)

def my_foward(x):
    h = l1(x)
    return l2(h)

# -- 動作検証 --
input_array = np.array([np.arange(1, 5)], dtype=np.float32)
x = Variable(input_array)
y = my_foward(x)
print(y.data)

# -- クラスで記述 --
class MyClass():
    def __init__(self):
        self.l1 = L.Linear(4, 3)
        self.l2 = L.Linear(3, 2)

    def foward(self, x):
        h = self.l1(x)
        return self.l2(h)

# -- 動作検証 --
input_array = np.array([np.arange(1, 5)], dtype=np.float32)
x = Variable(input_array)
my_class = MyClass()
y = my_class.foward(x)
print(y.data)

# -- Chainクラスを継承 --
# -- ニューラルネットワークの構成 --
# -- 4 - 3 - 2 のネットワーク
# -- call で順伝播 活性化関数を定義
class MyChain(Chain):
    def __init__(self):
        super().__init__(
            l1=L.Linear(4, 3),
            l2=L.Linear(3, 2),
        )

    def __call__(self, x):
        h = self.l1(x)
        return self.l2(h)

print()
# -- 動作検証 --
input_array = np.array([np.arange(1, 5)], dtype=np.float32)
x = Variable(input_array)
print(x.data)
print(x.data.shape)
my_chain = MyChain()
y = my_chain(x)
print(y.data)
print(y.data.shape)

