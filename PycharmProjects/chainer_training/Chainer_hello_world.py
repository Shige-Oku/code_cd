# coding: UTF-8

import chainer
from chainer import Variable, Chain, optimizers
import chainer.links as L
import chainer.functions as F
from chainer.functions.loss.mean_squared_error import mean_squared_error

import numpy as np
import matplotlib.pyplot as plt

# -- 検証用データの準備　階段関数のデータ --
# -- x:入力 t:正解値
x, t = [ ], [ ]
# -- linspace 開始、終了、作成数 --
for i in np.linspace(-1, 1, 100):
    x.append([i])
    if i < 0:
        t.append([0])
    else:
        t.append([1])

# plt.plot(np.array(x, dtype=np.float32).flatten(), np.array(t, dtype=np.float32).flatten())
# plt.show()

# -- Chainの記述 --
# -- ニューラルネットワークの構成 --
# -- 入力:1 中間ニューロン:10　出力:1 1-10-1 --
# -- call で順伝播 活性化関数（シグモイド関数）を定義
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            l1=L.Linear(1, 10),
            l2=L.Linear(10, 1),
        )

    def __call__(self, x):
        h1 = F.sigmoid(self.l1(x))
        return self.l2(h1)

# -- Variableの記述
x = Variable(np.array(x, dtype=np.float32))
t = Variable(np.array(t, dtype=np.float32))

# -- Chainの記述 --
model = MyChain()
# model = L.Classifier(MyChain(), lossfun=mean_squared_error)

# -- Optimizerの記述 --
# -- 最適化アルゴリズム：Adam
optimizer = chainer.optimizers.Adam()
# optimizer = optimizers.SGD()
# -- モデルをセット --
optimizer.setup(model)

# -- 学習 --
y = None
for i in range(100000):

    # -- 勾配をクリア --
    model.cleargrads()
    # -- 順伝播 --
    y = model(x)

    # -- 学習過程の表示
    if i % 10000 == 0:
        #  -- flatten():配列を一次元に変換 --
        plt.plot(x.data.flatten(), y.data.flatten())
        # plt.plot(np.ravel(x.data, order="C"), np.ravel(y.data, order="C"))
        # plt.plot(np.reshape(x.data, -1), np.reshape(y.data, -1))
        plt.title("i = " + str(i))
        plt.show()

    # -- 損失関数による誤差の計算、この場合は二乗和誤差 --
    loss = F.mean_squared_error(y, t)
    # -- 逆伝播 --
    loss.backward()

    # -- Optimizerによる重みとバイアスの更新 --
    optimizer.update()

# -- 結果の表示 --
plt.plot(x.data.flatten(), y.data.flatten())
plt.title("Finish!")
plt.show()
