# coding: UTF-8

import chainer
import numpy as np
from chainer import Variable, Chain, optimizers
import chainer.links as L
import chainer.functions as F

# -- モデルをクラスで記述 --
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(1, 2),
            l2=L.Linear(2, 1),
        )

    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        return self.l2(h)

# -- Optimizerの記述 --
model = MyChain()
optimizer = optimizers.SGD()  # 最適化アルゴリズム：確率的勾配法(SGD)
# optimizer = optimizers.Adam() # 最適化アルゴリズム：　Adam
optimizer.setup(model)

# -- Optimizerの実行 --
input_array = np.array([[1]], dtype=np.float32)
answer_array = np.array([[1]], dtype=np.float32)
x = Variable(input_array)
t = Variable(answer_array)

model.cleargrads()
y = model(x)

# -- 損失関数　二乗和誤差 --
loss = F.mean_squared_error(y, t)
# -- 誤差を逆伝播 --
loss.backward()

print()
print("update before")
print(model.l1.W.data)
print(model.l1.b.data)
print(model.l2.W.data)
print(model.l2.b.data)
optimizer.update()
print("update after")
print(model.l1.W.data)
print(model.l1.b.data)
print(model.l2.W.data)
print(model.l2.b.data)
