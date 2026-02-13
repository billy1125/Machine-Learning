import numpy as np

# 神經網路層類別
class Layer:
    def __init__(self):
        pass

    def forward(self, x):
        # 前向傳播，實際運算由子類別實作
        raise NotImplementedError

    def backward(self, grad):
        # 反向傳播，實際梯度計算由子類別實作
        raise NotImplementedError

class Dense(Layer):
    def __init__(self, input_dim, out_dim, activation=None):
        super().__init__()
        # 初始化權重矩陣，數值縮小避免一開始數值太大
        self.W = np.random.randn(input_dim, out_dim) * 0.01
        # 初始化 bias，全部設為 0
        self.b = np.zeros((1, out_dim))

        # 紀錄使用的啟動函數類型
        self.activation = activation
        # 儲存前向傳播後的輸出
        self.A = None

    def forward(self, x):
        # 前向傳播：計算線性組合 f(x) = xW + b
        self.x = x  # 保留輸入，供反向傳播使用
        Z = np.matmul(x, self.W) + self.b
        # 套用啟動函數
        self.A = self.g(Z)
        return self.A

    def backward(self, dA_out):
        # 反向傳播：計算梯度
        A_in = self.x  # 前一層的輸入
        # 計算 dZ
        dZ = self.dZ_(dA_out)

        # 計算權重與 bias 的梯度
        self.dW = np.dot(A_in.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=True)
        # 計算傳回前一層的梯度
        dA_in = np.dot(dZ, np.transpose(self.W))
        return dA_in

    def g(self, z):
        # 啟動函數
        if self.activation == 'relu':
            # ReLU
            return np.maximum(0, z)
        elif self.activation == 'Sigmoid':
            # Sigmoid
            return 1 / (1 + np.exp(-z))
        elif self.activation == 'Softmax':
            A = np.exp(z - np.max(z, axis=-1, keepdims=True))
            return A / np.sum(A, axis=-1, keepdims=True)
        else:
            # 沒有指定啟動函數就直接輸出
            return z

    def dZ_(self, dA_out):
        # 計算啟動函數對 Z 的微分
        if self.activation == 'relu':
            # ReLU 的梯度
            grad_g_z = 1. * (self.A > 0)
            return np.multiply(dA_out, grad_g_z)
        elif self.activation == 'Sigmoid':
            # Sigmoid 的梯度
            grad_g_z = self.A * (1 - self.A)
            return np.multiply(dA_out, grad_g_z)
        else:
            # 無啟動函數時，梯度直接往前傳
            return dA_out
        
class NeuralNetwork:
    def __init__(self):
        self._layers = []   # 用來存放所有網路層的清單

    def add_layer(self, layer):
        self._layers.append(layer)  # 將新的 layer 加到網路中

    def forward(self, X):
        self.X = X              # 記錄輸入資料
        for layer in self._layers:
            X = layer.forward(X)  # 依序經過每一層做 forward 計算
        return X                 # 回傳最後輸出

    def predict(self, X):
        p = self.forward(X)     # 先做 forward 得到預測結果

        if p.ndim == 1:         # 如果是單一樣本
            return np.argmax(p)  # 回傳機率最大的類別

        # 多筆樣本的情況
        return np.argmax(p, axis=1)


    def backward(self, loss_grad, reg=0.):
        # 反向傳播，從最後一層往前算梯度
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            loss_grad = layer.backward(loss_grad)

        # 將正則化項加入權重梯度
        for i in range(len(self._layers)):
            self._layers[i].dW += 2 * reg * self._layers[i].W

    def reg_loss(self, reg):
        loss = 0
        # 計算所有層的 L2 正則化損失
        for i in range(len(self._layers)):
            loss += reg * np.sum(self._layers[i].W * self._layers[i].W)
        return loss

    def update_parameters(self, learning_rate):
        # 依照學習率更新每一層的權重與偏置
        for i in range(len(self._layers)):
            self._layers[i].W += -learning_rate * self._layers[i].dW
            self._layers[i].b += -learning_rate * self._layers[i].db

    def parameters(self):
        params = []
        # 收集所有層的參數（權重與偏置）
        for i in range(len(self._layers)):
            params.append(self._layers[i].W)
            params.append(self._layers[i].b)
        return params

    def grads(self):
        grads = []
        # 收集所有層的梯度
        for i in range(len(self._layers)):
            grads.append(self._layers[i].dW)
            grads.append(self._layers[i].db)
        return grads