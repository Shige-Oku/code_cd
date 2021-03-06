# coding: UTF-8
import math

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

# ニューラルネットワークのインスタンス生成
neural_network = NeralNetwork()

# 実行
trial_data = [1.0, 2.0, 3.0]
print(neural_network.commit(trial_data))