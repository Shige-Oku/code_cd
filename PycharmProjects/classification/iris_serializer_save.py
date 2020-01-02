# coding: UTF-8

import chainer
from chainer import Variable, Chain, optimizers, serializers
import chainer.links as L
import chainer.functions as F

from chainer.datasets import tuple_dataset
from chainer import training, iterators
from chainer.training import extensions

import numpy as np
from sklearn import datasets

# -- Iris データ読み込み --
iris_data = datasets.load_iris()

x = iris_data.data.astype(np.float32) # Chainer は float32型
t = iris_data.target # 正解値
n = t.size # データ数

# -- 教師データの下処理 --
t_matrix = np.zeros(3 * n).reshape(n, 3).astype(np.float32)
for i in range(n):
    # 正解の種類の位置に1をセット
    t_matrix[i, t[i]] = 1.0

# -- 訓練用データとテスト用データ　半分を訓練用、残りをテストデータ --
indexes = np.arange(n)
# インデックスが奇数のデータを訓練用、偶数をテスト用に振り分け
indexes_train = indexes[indexes % 2 != 0]
indexes_test  = indexes[indexes % 2 == 0]

x_train = x[indexes_train, : ] # 訓練用　入力
t_train = t_matrix[indexes_train, : ] # 訓練用　正解
x_test  = x[indexes_test, : ] # テスト用　入力
t_test  = t_matrix[indexes_test, : ] # テスト用　正解

train = tuple_dataset.TupleDataset(x_train, t_train)

x_test_v = Variable(x_test)

# -- Chainの記述 --
class IrisChain(Chain):
    def __init__(self):
        # -- 入力 4 => 6 => 6 => 出力 3 --
        super(IrisChain, self).__init__(
            l1 = L.Linear(4, 6),
            l2 = L.Linear(6, 6),
            l3 = L.Linear(6, 3)
        )

    def __call__(self, x, t):
        # -- 損失関数：二乗和誤差 --
        return F.mean_squared_error(self.predict(x), t)

    def predict(self, x):
        # -- 活性化関数　中間：シグモイド関数、出力：恒等写像 --
        h1 = F.sigmoid(self.l1(x))
        h2 = F.sigmoid(self.l2(h1))
        h3 = self.l3(h2)
        return h3

# -- モデルとoptimizerの設定 --
model = IrisChain()
optimizer = optimizers.Adam()
optimizer.setup(model)

# -- trainerによる学習 --
# -- 30 個づつ一括りして返してくれる --
# -- dataset, batch_size, repeat, shuffle、orlder_sampler --
train_iter = iterators.SerialIterator(train, 30)
# -- iterator, optimizer, converter, device, loss_func --
# -- Updater の用意 --
updater = training.StandardUpdater(train_iter, optimizer)
# -- Trainer の用意 --
trainer = training.Trainer(updater, (5000, 'epoch'))
trainer.extend(extensions.ProgressBar())
trainer.run()

# -- モデルの保存 --
serializers.save_npz("my_iris.npz", model)

