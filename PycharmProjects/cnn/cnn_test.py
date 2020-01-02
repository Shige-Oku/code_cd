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

# print("Train:", len(train_data))
# print("Test:", len(test_data))

# -- MNIST画像の表示 --
#
# index = 2
# plt.imshow(train_data[index][0].reshape(28, 28), cmap='gray')
# plt.title(train_data[index][1])
# plt.show()

# -- Chain の記述 --
class MyMNIST(Chain):
    def __init__(self):
        super(MyMNIST, self).__init__(
            # L.Convolution2D（チャンネル数、フィルター数、フィルタサイズ）
            cnn1 = L.Convolution2D(1, 15, 5),  # 画像サイズ(1, 28, 28) → (15, 24, 24）
            cnn2 = L.Convolution2D(15, 40, 5), # 画像サイズ(15, 12, 12) → (40, 8, 8)
            l1 = L.Linear(640, 400),  # 入力は 40 × 4 × 4 = 640
            l2 = L.Linear(400, 10),
        )

    def __call__(self, x, t):
        # 出力：ソフトマックス関数　損失関数：交差エントロピー関数
        return F.softmax_cross_entropy(self.predict(x), t)

    def predict(self, x):
        # F.max_pooling_2d（入力画像、プーリング領域のサイズ）
        # 各中間層の活性化関数は Relu
        h1 = F.max_pooling_2d(F.relu(self.cnn1(x)), 2)   # 画像サイズ(15, 24, 24) → (15, 12, 12)
        h2 = F.max_pooling_2d(F.relu(self.cnn2(h1)), 2)  # 画像サイズ(40,  8,  8) → (40,  4,  4)
        # 全結合中間層ではドロップアウトする
        h3 = F.dropout(F.relu(self.l1(h2)))
        return self.l2(h3)

# -- モデルとoptimizerの設定 --
model = MyMNIST()
optimizer = optimizers.Adam()
optimizer.setup(model)

# -- 学習 --
# バッチサイズは 500
iterator = iterators.SerialIterator(train_data, 500)
updater  = training.StandardUpdater(iterator, optimizer)
trainer  = training.Trainer(updater, (20, 'epoch'))
trainer.extend(extensions.ProgressBar())
trainer.run()

# -- モデルの保存 --
serializers.save_npz("my_mnist.npz", model)

# -- テスト --
correct = 0
for i in range(len(test_data)):
    x = Variable(np.array([test_data[i][0]], dtype=np.float32))
    t = test_data[i][1]
    y = model.predict(x)
    maxIndex = np.argmax(y.data)
    if (maxIndex == t):
        correct += 1
# -- 正解率 --
print("Correct:", correct, "Total:", len(test_data), "Accuracy:", correct * 100 / len(test_data), "%")
