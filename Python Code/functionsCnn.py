import numpy as np
import math
from init_weights import * 

# =============================================================================
# Layer：所有層的基底類別（介面定義）
# =============================================================================
class Layer:
    def __init__(self):
        # params：此層的可訓練參數（例如 W、b、gamma、beta...）
        self.params = None

    def forward(self, x):
        """前向傳播：輸入 x，輸出此層計算結果"""
        raise NotImplementedError

    def backward(self, x, grad):
        """反向傳播：輸入 x 與上游梯度 grad，回傳對輸入的梯度"""
        raise NotImplementedError

    def reg_grad(self, reg):
        """加上正則項的梯度（預設不做任何事）"""
        pass

    def reg_loss(self, reg):
        """正則項損失（預設為 0）"""
        return 0.


# =============================================================================
# Dense：全連接層（線性層） Z = XW + b
# =============================================================================
# ---------- 加權和計算 ----------
class Dense(Layer):
    # Z = XW+b
    def __init__(self, input_dim, out_dim, init_method=None):
        super().__init__()

        # 權重與偏置
        self.W = np.empty((input_dim, out_dim))
        self.b = np.zeros((1, out_dim))

        # 初始化參數（依照指定方法）
        self.reset_parameters(init_method)

        # params / grads：常見框架設計，方便 optimizer 取用
        self.params = [self.W, self.b]
        self.grads = [np.zeros_like(self.W), np.zeros_like(self.b)]

    def reset_parameters(self, init_method=None):
        """
        參數初始化：
        - init_method 若有提供，格式為 (method_name, value)
        - 若無提供，預設使用 'kaiming_uniform', math.sqrt(5)
        注意：原碼使用 math，但原碼未 import math；依你的要求不修正。
        """
        if init_method is not None:
            method_name, value = init_method
        else:
            method_name, value = 'kaiming_uniform', math.sqrt(5)

        if method_name == "kaiming_uniform":
            kaiming_uniform(self.W, value)
            fan_in = self.W.shape[0]
            bound = 1 / math.sqrt(fan_in)
            self.b = np.random.uniform(-bound, bound, (self.b.shape))

        elif method_name == "kaiming_normal":
            kaiming_normal(self.W, value)
            fan_in = self.W.shape[0]
            std = 1 / math.sqrt(fan_in)
            self.b = np.random.normal(0, std, (self.b.shape))

        elif method_name == "xavier_uniform":
            xavier_uniform(self.W, value)

        elif method_name == "xavier_normal":
            xavier_normal(self.W, value)

        else:
            # 其他初始化策略：用標準常態分佈乘上 value
            self.W[:] = np.random.randn(self.W.shape) * value  # 0.01 * np.random.randn

    def forward(self, x):
        """
        forward：
        - 支援多維輸入（例如 NCHW），會先攤平成 (batch, features)
        """
        self.x = x
        # 將多通道/多維輸入攤平
        x1 = x.reshape(x.shape[0], np.prod(x.shape[1:]))
        Z = np.matmul(x1, self.W) + self.b
        return Z

    def __call__(self, x):
        return self.forward(x)

    def backward(self, dZ):
        """
        backward：
        - dZ：上游對 Z 的梯度
        - 回傳 dx：對輸入 x 的梯度（shape 會還原回原本 x.shape）
        """
        x = self.x
        x1 = x.reshape(x.shape[0], np.prod(x.shape[1:]))

        dW = np.dot(x1.T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)

        dx = np.dot(dZ, np.transpose(self.W))
        dx = dx.reshape(x.shape)  # 反攤平回原本輸入形狀

        # 原碼採「累加」方式存梯度（支援累積多次 backward）
        self.grads[0] += dW
        self.grads[1] += db

        return dx

    # -------- 添加正則項的梯度 --------
    def reg_grad(self, reg):
        # L2 正則：d/dW (reg * sum(W^2)) = 2*reg*W
        self.grads[0] += 2 * reg * self.W

    def reg_loss(self, reg):
        return reg * np.sum(self.W ** 2)

    def reg_loss_grad(self, reg):
        self.grads[0] += 2 * reg * self.W
        return reg * np.sum(self.W ** 2)


# =============================================================================
# Activation：Relu / Sigmoid / Tanh / Leaky ReLU
# =============================================================================
class Relu(Layer):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_output):
        # 若 x > 0，導數為 1；否則為 0
        x = self.x
        relu_grad = x > 0
        return grad_output * relu_grad


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        self.x = x
        return 1.0 / (1.0 + np.exp(-x))

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_output):
        # sigmoid'(x) = a*(1-a)
        x = self.x
        a = 1.0 / (1.0 + np.exp(-x))
        return grad_output * a * (1 - a)


class Tanh(Layer):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        self.x = x
        self.a = np.tanh(x)
        return self.a

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_output):
        # tanh'(x) = 1 - tanh(x)^2
        d = (1 - np.square(self.a))
        return grad_output * d


class Leaky_relu(Layer):
    def __init__(self, leaky_slope):
        super().__init__()
        self.leaky_slope = leaky_slope

    def forward(self, x):
        self.x = x
        return np.maximum(self.leaky_slope * x, x)

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_output):
        x = self.x
        d = np.zeros_like(x)
        d[x <= 0] = self.leaky_slope
        d[x > 0] = 1
        return grad_output * d
