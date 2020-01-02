# coding:UTF-8
from sklearn import datasets
from sklearn import svm
# 文字認識正解率の表示
from sklearn import metrics
import matplotlib.pyplot as plt

# 数字データの読み込み
digits = datasets.load_digits()

# データの形式を確認
print(digits.data)
print(digits.data.shape)

# データの数
n = len(digits.data)
print(len(digits.data))

# 画像と正解値の表示
# images = digits.images
# labels = digits.target
# for i in range(10):
#     # 複数のグラフを表示　2行、5列、表示位置
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(images[i], cmap=plt.cm.gray_r, interpolation="nearest")
#     # 軸表示なし
#     plt.axis("off")
#     plt.title("Training: " + str(labels[i]))
# plt.show()

# サポートベクターマシン
# ganmma：１つの訓練データの影響の大きさ　C：ご認識の許容度
clt = svm.SVC(gamma=0.001, C=100.0)
# サポートベクターマシンによる訓練（６割のデータを使用、残りの４割は検証用）
clt.fit(digits.data[:int(n*6/10)], digits.target[:int(n*6/10)])

# 最後の10個のデータをチェック
# 正解（マイナスを指定すると末尾からの範囲）
# print(digits.target[-10:])
# 予測を行う（数字を読み取る）
# print(clt.predict(digits.data[-10:]))

# 残り４割の画像から、数字を読み取る
# 正解
expected = digits.target[int(-n*4/10):]
# 予測
predicted = clt.predict(digits.data[int(-n*4/10):])
# 正解率
print(metrics.classification_report(expected, predicted))
# 誤認識のマトリックス
print(metrics.confusion_matrix(expected, predicted))

# 予測と画像の対応（一部）
images = digits.images[int(-n*4/10):]
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.axis("on")
    plt.imshow(images[i], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title("Guess: " + str(predicted[i]))
plt.show()

# 学習した６割の画像でテスト
# 正解
expected2 = digits.target[:int(n*6/10)]
# 予測
predicted2 = clt.predict(digits.data[:int(n*6/10)])
# 正解率
print(metrics.classification_report(expected2, predicted2))
# 誤認識のマトリックス
print(metrics.confusion_matrix(expected2, predicted2))

# 予測と画像の対応（一部）
images2 = digits.images[:int(n*6/10)]
for i in range(12):
    plt.subplot(3, 4, i + 1)
    plt.axis("on")
    plt.imshow(images2[i], cmap=plt.cm.gray_r, interpolation="nearest")
    plt.title("Guess: " + str(predicted2[i]))
plt.show()

# TODO : metrics, classification_report, confusion_matrix 確認
