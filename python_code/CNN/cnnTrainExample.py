# import util

import numpy as np
import functionsAnn as fn
import classTrain as train

from classCnn import NeuralNetwork
from classAnn import *
from functionsCnn import *


# 固定隨機種子，確保每次訓練的初始化權重一致，方便實驗比對
np.random.seed(1)

# 初始化神經網路容器
nn = NeuralNetwork()

# 搭建網路架構：
# 第一層全連接層 (Dense)：輸入 2 維 (x, y 座標)，輸出 100 維
nn.add_layer(Dense(2, 100))
# 激發函數 (Activation)：使用 Relu 增加模型的非線性表達能力
nn.add_layer(Relu())
# 第二層全連接層 (輸出層)：將 100 維映射到 3 個類別 (對應 y 的種類)
nn.add_layer(Dense(100, 3)) 

# 產生測試資料：X 為座標點，y 為標籤 (0, 1, 2)
X, y = fn.gen_spiral_dataset(N=100, D=2, K=3)

# 設定優化器 (Optimizer)：使用帶有動量 (Momentum) 的隨機梯度下降
# nn.parameters() 會將網路中所有層的權重與梯度交給優化器管理
learning_rate = 1e-1
momentum = 0.9
optimizer = train.SGD(nn.parameters(), learning_rate, momentum)

# 設定超參數 (Hyperparameters)
epochs = 5000         # 總訓練輪數
batch_size = len(X)   # 批次大小，這裡使用 Full Batch 訓練
reg = 0.5e-3          # 正規化係數 (L2 Regularization)，避免權重過大導致過擬合
print_n = 300         # 每隔多少輪印出一次損失值 (Loss)

# 開始訓練：執行前向傳播、計算誤差、反向傳播並更新參數
losses = train.train_nn(nn, X, y, optimizer, fn.cross_entropy_grad_loss, 
                        epochs, batch_size, reg, print_n)

# 評估模型：計算訓練後的準確度 (Accuracy)
print(np.mean(nn.predict(X) == y))