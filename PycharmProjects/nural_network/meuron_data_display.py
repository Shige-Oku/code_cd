# coding: UTF-8
import math
import matplotlib.pyplot as plt

# シグモイド関数
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

# ニューロン
class Neuron:
    input_sum = 0.0
    output = 0.0

    def setInput(self, inp):
        self.input_sum += inp
        # print(self.input_sum)

    def getOutput(self):
        self.output = sigmoid(self.input_sum)
        return self.output

# ニューラルネットワーク
class NeralNetwork:
    # 入力の重み
    w = [1.5, -2.5, -0.5]
    # ニューロンのインスタンス生成
    neuron = Neuron()
    # 実行
    def commit(self, input_data):
        for cnt in range(len(input_data)):
            self.neuron.setInput(input_data[cnt] * self.w[cnt])
        return self.neuron.getOutput()

# 基準点（データの範囲を0.0 - 1.0の範囲に収めるため）
refer_point_0 = 34.5
refer_point_1 = 137.5

# ファイルの読みこみ
trial_data = []
with open("trial_data", "r") as f:
    for line in f:
        line = line.rstrip().split(",")
        trial_data.append([float(line[0]) - refer_point_0, float(line[1]) - refer_point_1])

# ニューラルネットワークのインスタンス生成
neural_network = NeralNetwork()

# 実行
position = [[], []]
for data in trial_data:
    position[0].append(data[1] + refer_point_1)
    position[1].append(data[0] + refer_point_0)

# プロット
plt.scatter(position[0], position[1], c="red", label="Position", marker="+")

plt.legend()
plt.show()