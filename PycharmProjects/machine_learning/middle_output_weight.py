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
        # print("input_sum:" + str(self.input_sum))
        return self.output

    def reset(self):
        self.input_sum = 0
        self.output = 0

# ニューラルネットワーク
class NeralNetwork:
    # 入力の重み
    # 入力層と中間層　入力１と中間層、入力２と中間層、中間層のバイアス
    w_im = [[0.496, 0.512], [-0.501, 0.998], [0.498, -0.502]]
    # 中間層と出力層　中間層１と出力層、中間層２と出力層　出力層のバイアス
    w_mo = [0.121, -0.4996, 0.200]

    # 各層の宣言
    # 入力層　入力値、入力値、バイアス
    input_layer = [0.0, 0.0, 1.0]
    # ニューロンのインスタンス生成
    # 中間層　第一ノード、第２ノード、バイアス
    middle_layer = [Neuron(), Neuron(), 1.0]
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
        sou = 0
        while sou < len(self.input_layer)-1: # バイアスの分は回さない
            cul = 0
            while cul < len(self.w_im):
                self.middle_layer[sou].setInput(self.input_layer[cul] * self.w_im[cul][sou])
                cul += 1
            sou += 1

        # 中間層→出力層
        cul_o = 0
        while cul_o < len(self.middle_layer)-1: # バイアスの分は回さない
            self.output_layer.setInput(self.middle_layer[cul_o].getOutput() * self.w_mo[cul_o])
            cul_o += 1
        self.output_layer.setInput(self.middle_layer[2] * self.w_mo[2])

        return self.output_layer.getOutput()

    def learn(self, input_data):
        print(input_data)

        # 出力層 緯度、経度
        output_data = self.commit([input_data[0], input_data[1]])
        # 正解値
        correct_value = input_data[2]
        # 学習係数
        k = 0.3

        # 出力層=>中間層　正解値との誤差 * 出力の微分値（シグモイド出力 * (1 - シグモイド出力））
        delta_w_mo = (correct_value - output_data) * output_data * (1.0 - output_data)
        # w_mo：中間層と出力層の重み。後で入力層と中間層の重みを更新する際に使用する。
        old_w_mo = list(self.w_mo)
        # 重みの更新
        self.w_mo[0] += self.middle_layer[0].output * delta_w_mo * k
        self.w_mo[1] += self.middle_layer[1].output * delta_w_mo * k
        self.w_mo[2] += self.middle_layer[2] * delta_w_mo * k

# 基準点（データの範囲を0.0 - 1.0の範囲に収めるため）
refer_point_0 = 34.5
refer_point_1 = 137.5

# ファイルの読みこみ
training_data = []
with open("training_data", "r") as f:
    for line in f:
        # rstirp()：行末の改行コード削除　split()：指定文字で分割
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
