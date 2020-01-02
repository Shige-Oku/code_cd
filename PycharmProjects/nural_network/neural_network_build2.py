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

    # 入力値を加算する
    def setInput(self, cinp):
        self.input_sum += inp
        # print(self.input_sum)

    # 入力値をシグモイド関数を使って評価
    def getOutput(self):
        self.output = sigmoid(self.input_sum)
        # print("input_sum:" + str(self.input_sum))
        return self.output

    # 次のデータのためリセットする
    def reset(self):
        self.input_sum = 0
        self.output = 0

    def getInput(self):
        return self.input_sum

# ニューラルネットワーク
class NeralNetwork:
    # 入力の重み
    # 入力層と中間層　入力１と中間層、入力２と中間層、中間層のバイアス
    w_im = [[0.496, 0.512], [-0.501, 0.998], [0.498, -0.502]]
    # 中間層と出力層　中間層１と出力層、中間層２と出力層　出力層のバイアス
    w_io = [0.121, -0.4996, 0.200]

    # 各層の宣言
    # 入力層　入力値、入力値、バイアス
    input_layer = [0.0, 0.0, 1.0]
    # ニューロンのインスタンス生成
    # 中間層　第一ノード、第２ノード、バイアス
    middle_layer = [Neuron(), Neuron(), 1.0]c
    # 出力層
    output_layer = Neuron()

    # 実行
    def commit(self, input_data):
        # 各層のリセット
        # 入力層の初期化
        self.input_layer[0] = input_data[0]
        self.input_layer[1] = input_data[1]

        # 中間層のリセット
        self.middle_layer[0].reset()
        self.middle_layer[1].reset()

        # 出力層のリセット
        self.output_layer.reset()

        # 入力層→中間層
        # self.middle_layer[0].setInput(self.input_layer[0] * self.w_im[0][0])
        # self.middle_layer[0].setInput(self.input_layer[1] * self.w_im[1][0])
        # self.middle_layer[0].setInput(self.input_layer[2] * self.w_im[2][0])
        #
        # self.middle_layer[1].setInput(self.input_layer[0] * self.w_im[0][1])
        # self.middle_layer[1].setInput(self.input_layer[1] * self.w_im[1][1])
        # self.middle_layer[1].setInput(self.input_layer[2] * self.w_im[2][1])

        sou = 0
        # print(self.middle_layer[0].getInput())
        # print(self.middle_layer[1].getInput())

        while sou < len(self.input_layer)-1: # バイアスの分は回さない
            cul = 0
            while cul < len(self.w_im):
                self.middle_layer[sou].setInput(self.input_layer[cul] * self.w_im[cul][sou])
                cul += 1
            sou += 1

        # 中間層→出力層
        # self.output_layer.setInput(self.middle_layer[0].getOutput() * self.w_io[0])
        # self.output_layer.setInput(self.middle_layer[1].getOutput() * self.w_io[1])
        # self.output_layer.setInput(self.middle_layer[2] * self.w_io[2])
        cul_o = 0
        while cul_o < len(self.middle_layer)-1: # バイアスの分は回さない
            self.output_layer.setInput(self.middle_layer[cul_o].getOutput() * self.w_io[cul_o])
            cul_o += 1
        self.output_layer.setInput(self.middle_layer[2] * self.w_io[2])

        return self.output_layer.getOutput()

# 基準点（データの範囲を0.0 - 1.0の範囲に収めるため）
refer_point_0 = 34.5
refer_point_1 = 137.5

# ファイルの読みこみ
trial_data = []
with open("trial_data", "r") as f:
    for line in f:
        line = line.rstrip().split(",")
        trial_data.append([float(line[0]) - refer_point_0, float(line[1]) - refer_point_1])
# print(trial_data)

# ニューラルネットワークのインスタンス生成
neural_network = NeralNetwork()

# 実行緯度
position_tokyo = [[], []]
position_kanagawa = [[], []]
for data in trial_data:
    return_value = neural_network.commit(data)
    # print(return_value)
    # print("keido:" + str(data[1]) + ", ido:" + str(data[0]))
    if return_value < 0.5:
        position_tokyo[0].append(data[1] + refer_point_1)
        position_tokyo[1].append(data[0] + refer_point_0)
    else:
        position_kanagawa[0].append(data[1] + refer_point_1)
        position_kanagawa[1].append(data[0] + refer_point_0)

# プロット
plt.scatter(position_tokyo[0], position_tokyo[1], c="red", label="Tokyo", marker="+")
plt.scatter(position_kanagawa[0], position_kanagawa[1], c="blue", label="kanagawa", marker="+")

plt.legend()
plt.show()
