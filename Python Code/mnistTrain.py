import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import functionsAnn as funann
import classAnn as classann

current_path = Path(__file__).resolve().parent
target_path = current_path.parent / 'Data' / 'mnist_train.csv'

data = pd.read_csv(target_path)
print(data.head())
print(data.label.unique())
print(data.describe())
print(data.info())

# 1. One-Hot 轉換
y_onehot_df = pd.get_dummies(data['label'])
y = y_onehot_df.values.astype(int) # 此時 y 的 shape 為 (150, 3)
print(y[:5])

# 2. 取得特徵
X = data.iloc[:, 1:].values  # 直接轉成 numpy array 供模型使用
X = X / 255.0

# 設定訓練參數
epochs = 5            # 總共要跑多少個循環（Epochs）
learning_rate = 0.5     # 學習率（步長），這裡設定為 1.0
batch_size = 50         # 每個批次的資料筆數
reg = 1e-4                # 正規化係數（Regularization Strength），用來防止過擬合（Overfitting）
print_n = epochs * len(X) // 32 // 10    # 設定每跑 10% 的進度就印一次 Log


ann = classann.NeuralNetwork()

ann.add_layer(classann.Dense(784, 200, 'relu'))
ann.add_layer(classann.Dense(200, 100, 'relu'))
ann.add_layer(classann.Dense(100, 10))

funann.train_batch(ann, X, y, funann.cross_entropy_grad_loss, epochs, batch_size, learning_rate, reg, print_n, False, True)

# 訓練完成後，計算模型在訓練集上的準確度（Accuracy）
# nn.predict(X) 會回傳預測類別，並與真實標籤 y 比較取平均值
# print(np.mean(ann.predict(X) == y))

accuracy = np.mean(ann.predict(X) == data['label'].values)
print(f"訓練集準確度: {accuracy}")