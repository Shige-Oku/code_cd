# coding: UTF-8

import chainer
import numpy as np
from chainer import Variable

# -- numpy の配列を作成 --
input_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
print(input_array)
print(input_array.shape)

# -- Valiableオブジェクトを作成 --
# 入力の形式（数）、入力値の保存　入力の勾配など
x = Variable(input_array)
print(x.data)

# -- 計算 --
# yもVariableオブジェクト
# y = x * 2 + 1
y = x ** 2 + 2 * x + 1 # 微分形はy' = 2x + 2
print(y.data)
print()

# -- 微分値を求める --
# 要素が複数の場合はy.dataに初期値が必要
# とりあえず y の勾配を 1とする　出力 y の勾配から入力 x の勾配を求める
y.grad = np.ones((2, 3), dtype=np.float32)
# y→xと遡って微分値を求める
y.backward()
# y = x * 2 + 1の微分形は y = 2
print(x.grad)
print(y.grad)
print()

# print(np.arange(1, 6))
x2 = Variable(np.array([np.arange(1, 6), np.arange(6, 11)], dtype=np.float32))
print(x2.data)
y2 = 3 * (x2 ** 2) + 2 * x2 + 1 # 微分形は y' = 6 * x + 2
print(y2.data)
y2.grad = np.ones((2, 5), dtype=np.float32)
y2.backward()
print(x2.grad)
print()

x3 = Variable(np.array([np.arange(1, 4), np.arange(4, 7), np.arange(7, 10)], dtype=np.float32))
print(x3)
y3 = x3 ** 3 + 2 * (x3 ** 2) + 3 * x3 + 1 # 微分形は y' = 3 * X ** 2 + 4 * x + 3
print(y3)
y3.grad = np.ones((3, 3), dtype=np.float32)
y3.backward()
print(x3.grad)

