# coding: UTF-8

import chainer
from chainer import Function, gradient_check, report, training, utils
from chainer import Variable, Chain, optimizers, serializers
from chainer import Link, Chain, ChainList
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

    def __call__(self, x):
        # -- 活性化関数　中間：ランプ関数、出力：恒等写像 --
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        h3 = self.l3(h2)
        return h3

# -- モデルとoptimizerの設定 --
model = IrisChain()
# -- 引数に損失関数を指定しない場合は softmax_cross_entropy --
model = L.Classifier(model)
optimizer = optimizers.Adam()
optimizer.setup(model)

# -- Trainerによる学習 --
# -- 30 個づつ一括りして返してくれる --
# -- dataset, batch_size, repeat, shuffle、orlder_sampler --
train_iter = iterators.SerialIterator(train, 30)
# -- Updater の用意 --
# -- iterator, optimizer, converter, device, loss_func --
updater = training.StandardUpdater(train_iter, optimizer)
# -- Trainer の用意 --
trainer = training.Trainer(updater, (5000, 'epoch'))
trainer.extend(extensions.ProgressBar())
trainer.extend(extensions.PlotReport(['main\loss', 'validation\main'], x_key='epoch', file_name='loss_png'))
trainer.extend(extensions.PlotReport(['main\accuracy', 'validation\main\accuracy'], x_key='epoch', file_name='accuracy.png'))
trainer.run()

# -- テスト --
model.cleargrads()
# y_test_v = model.predict(x_test_v)
y_test_v = model(x_test_v)
y_test = y_test_v.data

# -- 正解数のカウント --
correct = 0
rowCount = y_test.shape[0]

for i in range(rowCount):
    # np.argmax 関数は最大の要素のインデクスを返す --
    maxIndex = np.argmax(y_test[i, :])
    print(y_test[1, :], maxIndex)
    # if maxIndex == t_test[i]:
    if t_test[i, maxIndex] == 1:
        correct += 1

# -- 正解率 --
print("Correct:", correct, "Total:", rowCount, "Acurracy:", correct / rowCount * 100, "%")
