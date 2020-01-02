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
        print("input_sum:" + str(self.input_sum))
        return self.output

    def reset(self):
        self.input_sum = 0
        self.output = 0

# ニューラルネットワーク
class NeralNetwork:
    # 入力の重み
    w = [-0.5, 0.5, -0.2]
    # ニューロンのインスタンス生成
    neuron = Neuron()
    # 実行
    def commit(self, input_data):
        self.neuron.reset()

        bias = 1.0

        for cnt in range(len(input_data)):
            self.neuron.setInput(input_data[cnt] * self.w[cnt])
        # バイアス
        self.neuron.setInput(bias * self.w[2])
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
print(trial_data)

# ニューラルネットワークのインスタンス生成
neural_network = NeralNetwork()

# 実行
position_tokyo = [[], []]
position_kanagawa = [[], []]
for data in trial_data:
    return_value = neural_network.commit(data)
    print(return_value)
    print("keido:" + str(data[1]) + ", ido:" + str(data[0]))
    if return_value < 0.5:
        position_tokyo[0].append(data[1] + refer_point_1)
        position_tokyo[1].append(data[0] + refer_point_0)
    else:
        position_kanagawa[0].append(data[1] + refer_point_1)
        position_kanagawa[1].append(data[0] + refer_point_0)

# print(position_tokyo[0])
# print(position_tokyo[1])
# print(position_kanagawa[0])
# print(position_kanagawa[1])

# プロット
plt.scatter(position_tokyo[0], position_tokyo[1], c="red", label="Tokyo", marker="+")
plt.scatter(position_kanagawa[0], position_kanagawa[1], c="blue", label="kanagawa", marker="+")

plt.legend()
plt.show()

# 東京都神奈川にきれいに分類できない => OK commit()でreset()していなかった。input_sumが先頭行からの合算になっていた。
# TODO なぜ重みが -0.5, 0.5 なのか =>
# 基準点からの差で、
# 経度が緯度より基準点から離れている場合に東京（マイナスで発火しない）、
# 緯度が離れているときが神奈川（プラスで発火）