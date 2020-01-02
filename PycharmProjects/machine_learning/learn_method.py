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
    # 入力層と中間層　入力層１と中間層、入力層２と中間層、中間層のバイアス
    w_im = [[0.496, 0.512], [-0.501, 0.998], [0.498, -0.502]]
    # 中間層と出力層　中間層１と出力層、中間層２と出力層　出力層のバイアス
    w_io = [0.121, -0.4996, 0.200]

    # 各層の宣言
    # 入力層　入力値、入力値、バイアス
    input_layer = [0.0, 0.0, 1,0]
    # ニューロンのインスタンス生成
    # 中間層　第一層、第２層、バイアス
    middle_layer = [Neuron(), Neuron(), 1.0]
    # 出力層
    output_layer = Neuron()

    # 実行
    def commit(self, input_data):
        # 各層のリセット
        # 入力層を入力値で初期化
        for in1 in range(len(input_data)):
            self.input_layer[in1] = input_data[in1]

        # 中間層初期化
        self.middle_layer[0].reset()
        self.middle_layer[1].reset()

        # 出力層初期化
        self.output_layer.reset()

        # 入力層=>中間層

        for cnt in range(len(input_data)):
            self.neuron.setInput(input_data[cnt] * self.w[cnt])
        # バイアス
        self.neuron.setInput(bias * self.w[2])
        return self.neuron.getOutput()

    def learn(self, input_data):
        print(input_data)

# 基準点（データの範囲を0.0 - 1.0の範囲に収めるため）
refer_point_0 = 34.5
refer_point_1 = 137.5

# ファイルの読みこみ
training_data = []
with open("training_data", "r") as f:
    for line in f:
        line = line.rstrip().split(",")
        # 緯度から基準点を減算、経度から基準点を減算、正解値
        training_data.append([float(line[0]) - refer_point_0, float(line[1]) - refer_point_1, int(line[2])])

# print(training_data)

# ニューラルネットワークのインスタンス生成
neural_network = NeralNetwork()

# 学習
neural_network.learn(training_data[0])
print(len(training_data))

# 訓練データの表示の準備
position_tokyo_learning = [[], []]
position_kanagawa_learning = [[], []]
for data in training_data:
    if data[2] < 0.5:
        position_tokyo_learning[0].append(data[1]+refer_point_1)
        position_tokyo_learning[1].append(data[0]+refer_point_0)
    else:
        position_kanagawa_learning[0].append(data[1]+refer_point_1)
        position_kanagawa_learning[1].append(data[0]+refer_point_0)

# プロット
plt.scatter(position_tokyo_learning[0], position_tokyo_learning[1], c="red", label="Tokyo learn", marker="+")
plt.scatter(position_kanagawa_learning[0], position_kanagawa_learning[1], c="blue", label="Kanagawa learn", marker="+")

plt.legend()
plt.show()
