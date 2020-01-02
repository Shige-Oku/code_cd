# coding: UTF-8

import chainer
import numpy as np
from chainer import Variable, Chain
import chainer.links as L
import chainer.functions as F

# -- 各値や活性化関数の呼び出し
sample_array = np.array([np.arange(1, 4)], dtype=np.float32)
x = Variable(sample_array)

y1 = F.sum(x)
print(y1.data)

y2 = F.average(x)
print(y2.data)

y3 = F.max(x)
print(y3.data)

y4 = F.sigmoid(x)
print(y4.data)

y5 = F.relu(x)
print(y5.data)