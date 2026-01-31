import numpy as np
import matplotlib.pyplot as plt

import functionsAnn as funann
import classAnn as classann

def train(nn, X, y, loss_function, epochs=10000, learning_rate=1e-0, reg=1e-3, print_n=10):
    # 開始執行指定次數的訓練循環 (Epochs)
    for epoch in range(epochs):
        # 1. 前向傳播 (Forward Pass)：將資料丟進網路，得到預測結果 f (Logits)
        f = nn.forward(X)

        # 2. 計算損失與梯度：使用傳入的 loss_function 算出 Loss 以及對應的梯度 loss_grad
        loss, loss_grad = loss_function(f, y)

        # 3. 加入正規化損失 (Regularization Loss)：防止權重過大導致過擬合 (Overfitting)
        loss += nn.reg_loss(reg)

        # 4. 後向傳播 (Backward Pass)：將梯度傳回網路各層，計算權重與偏置的梯度
        nn.backward(loss_grad, reg)

        # 5. 更新參數 (Parameter Update)：根據算出的梯度與學習率，調整網路參數
        nn.update_parameters(learning_rate)

        # 6. 定期列印 Log：每隔 print_n 次就印出目前的損失值，追蹤訓練進度
        if epoch % print_n == 0:
            print("iteration %d: loss %f" % (epoch, loss))

# 設定隨機編號（Seed），確保每次執行結果都一樣，方便除錯（Debug）
np.random.seed(89)

# 1. 產生螺旋資料
X, y = funann.gen_spiral_dataset(N=100, D=2, K=3)

# 2. 建立畫布
plt.figure(figsize=(8, 6))

# 3. 畫出散佈圖 (Scatter Plot)
# X[:, 0] 是橫軸 (x 座標), X[:, 1] 是縱軸 (y 座標)
# c=y 代表根據類別自動上色, s=40 是點的大小, edgecolor='k' 是加上黑色邊框比較好辨識
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral, edgecolor='k')

# 4. 加上標題與標籤
plt.title("Spiral Dataset Visualization")
plt.xlabel("Feature 1 (x)")
plt.ylabel("Feature 2 (y)")

# 5. 顯示圖表
plt.show()

# 設定訓練參數
epochs = 10000            # 總共要跑多少個循環（Epochs）
learning_rate = 1e-0      # 學習率（步長），這裡設定為 1.0
reg = 1e-4                # 正規化係數（Regularization Strength），用來防止過擬合（Overfitting）
print_n = epochs // 10    # 設定每跑 10% 的進度就印一次 Log

# 建立一個神經網路物件
ann = classann.NeuralNetwork()

# 新增第一層全連接層：輸入 2 維，輸出 100 維，使用 ReLU 當啟動函數
ann.add_layer(classann.Dense(2, 100, 'relu'))

# 新增第二層全連接層：輸入 100 維，輸出 3 維，使用 softmax 做分類
ann.add_layer(classann.Dense(100, 3))

# 開始訓練模型
# 傳入模型物件 nn、資料、損失與梯度的整合函式，以及相關超參數
train(ann, X, y, funann.cross_entropy_grad_loss, epochs, learning_rate, reg, print_n)

# 訓練完成後，計算模型在訓練集上的準確度（Accuracy）
# nn.predict(X) 會回傳預測類別，並與真實標籤 y 比較取平均值
print(np.mean(ann.predict(X) == y))