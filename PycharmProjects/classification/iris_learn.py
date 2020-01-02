# coding: UTF-8

import chainer
from chainer import Variable, Chain, optimizers
import chainer.links as L
import chainer.functions as F

import numpy as np
from sklearn import datasets

# -- Iris データ読み込み --
iris_data = datasets.load_iris()

x = iris_data.data.astype(np.float32)
t = iris_data.target
n = t.size

# -- 教師データの下処理 --
t_matrix = np.zeros(3 * n).reshape(n, 3).astype(np.float32)
for i in range(n):
    t_matrix[i, t[i]] = 1.0

# -- 訓練用データとテスト用データ　半分を訓練用、残りをテストデータ --
indexes = np.arange(n)
indexes_train = indexes[indexes % 2 != 0]
indexes_test  = indexes[indexes % 2 == 0]

x_train = x[indexes_train, : ] # 訓練用　入力
t_train = t_matrix[indexes_train, : ] # 訓練用　正解
x_test  = x[indexes_test, : ] # テスト用　入力
t_test  = t_matrix[indexes_test, : ] # テスト用　正解

x_train_v = Variable(x_train)
t_train_v = Variable(t_train)
x_test_v  = Variable(x_test)

# -- Chainの記述 --
class IrisChain(Chain):
    def __init__(self):
        super(IrisChain, self).__init__(
            l1 = L.Linear(4, 6),
            l2 = L.Linear(6, 6),
            l3 = L.Linear(6, 3)
        )

    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        h3 = F.self.l3(h2)
        return h3

# -- モデルとoptimizerの設定 --
model = IrisChain()
optimizer = optimizers.Adam()
optimizer.setup(model)

# -- 学習 --
for i in range(10000):

    # -- 勾配の初期化 --
    model.cleargrads()
    y_train_v = model(x_train_v)

    # -- 損失関数（二乗和誤差）による誤差の計算 --
    loss = F.mean_squared_error(y_train_v, x_train_v)
    loss.backward()

    # -- Optimizer による重みの更新 --
    optimizer.update(())
