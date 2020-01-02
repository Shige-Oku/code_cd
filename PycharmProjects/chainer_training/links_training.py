# coding: UTF-8

import chainer
import numpy as np
from chainer import Variable
import chainer.links as L

# -- LinksのLinear Link（入力に係数をかけたものと、バイアスをすべて足し合わせる）関数によりオブジェクトを作成 --
# 係数（重み）の初期値はランダム、バイアスの初期値は0
# l = L.Linear(3, 2)
l = L.Linear(4, 3)
print(l.W.data)
print(l.b.data)
print()

# -- オブジェクトlによりyを計算 --
# input_array = np.array([[1, 2, 3]], dtype=np.float32)
input_array = np.array([np.arange(1, 5), np.arange(5, 9), np.arange(9, 13)], dtype=np.float32)
# input_array = np.array([np.arange(1, 5), np.arange(5, 9)], dtype=np.float32)
print(input_array)
x = Variable(input_array)
y = l(x)
print(y.data)
print()

# -- lの勾配をゼロに初期化 --
l.cleargrads()

# -- y→lと遡って微分の計算  --
# y.grad = np.ones((1, 2), dtype=np.float32)
y.grad = np.ones((3, 3), dtype=np.float32)
y.backward()
print(l.W.grad)
print(l.W.grad.shape)
print(l.b.grad)
print(l.b.grad.shape)
