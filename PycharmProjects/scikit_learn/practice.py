# coding: UTF-8
from sklearn import datasets
from sklearn import svm

# データセットからIrisの測定データの読み込み
iris = datasets.load_iris()

# データの形式を確認
# print(iris.data)
# print(iris.data.shape)

# データの数
print(len(iris.data))

# 線形サポートベクターマシン
clf = svm.LinearSVC()
# サポートベクターマシンによる訓練
# irisのデータ、irisの正解値（品種）
clf.fit(iris.data, iris.target)

# 品種の判定
# Sepal（ガク）高さ、Sepalの幅、Petal（花弁）の高さ、Petal（花弁）の幅
print(clf.predict([[5.1, 3.5, 1.4, 0.1], [6.5, 2.5, 4.4, 1.4], [5.9, 3.0, 5.2, 1.5]]))
