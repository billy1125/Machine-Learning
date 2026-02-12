# -*- coding: utf-8 -*-
"""
整理重點（不改邏輯、不修錯、不中途刪任何程式碼）：
1) 保留原本所有程式碼與先後順序（避免影響執行結果）。
2) 只做「排版整理、補註解、統一註解為繁體（台灣用語）」。
3) 原碼中有一些重複定義（例如 one_hot、重複 import numpy），我不移除，會用註解標清楚。
"""

# =============================================================================
# 匯入套件 / 依賴模組
# =============================================================================
import numpy as np
from im2row import *          # 依賴：im2row / row2im 相關函式（外部檔案）
from init_weights import *    # 依賴：權重初始化相關函式（外部檔案）


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


# =============================================================================
# BatchNorm_1d：一維 Batch Normalization（以 (N, D) 的概念處理）
# =============================================================================
class BatchNorm_1d(Layer):
    def __init__(self, num_features, gamma_beta_method=None, eps=1e-8, momentum=0.9):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        # gamma/beta：若沒指定初始化方式，預設 gamma=1、beta=0
        if not gamma_beta_method:
            self.gamma = np.ones((1, num_features))
            self.beta = np.zeros((1, num_features))
        else:
            self.gamma = np.random.randn(1, num_features)
            self.beta = np.random.randn(1, num_features)  # 原碼備註保留

        self.params = [self.gamma, self.beta]
        self.grads = [np.zeros_like(self.gamma), np.zeros_like(self.beta)]

        # 推論時使用的滑動平均
        self.running_mu = np.zeros((1, num_features))
        self.running_var = np.zeros((1, num_features))

    def forward(self, X, training=True):
        if training:
            self.n_X = X.shape[0]
            self.X_shape = X.shape

            # 攤平成 (N, D)
            self.X_flat = X.ravel().reshape(self.n_X, -1)

            self.mu = np.mean(self.X_flat, axis=0)
            self.var = np.var(self.X_flat, axis=0)
            self.X_hat = (self.X_flat - self.mu) / np.sqrt(self.var + self.eps)

            out = self.gamma * self.X_hat + self.beta

            # 更新 running mean/var（原碼用區域變數接住，但不回寫 self.running_*）
            running_mu, running_var, momentum = self.running_mu, self.running_var, self.momentum
            running_mu = momentum * running_mu + (1 - momentum) * self.mu
            running_var = momentum * running_var + (1 - momentum) * self.var

        else:
            X_flat = X.ravel().reshape(X.shape[0], -1)
            # 正規化（注意：原碼用 eps 變數，但此分支沒有定義 eps；依要求不修正）
            X_hat = (X_flat - self.running_mu) / np.sqrt(self.running_var + eps)
            out = self.gamma * X_hat + self.beta

        return out.reshape(self.X_shape)

    def __call__(self, X):
        return self.forward(X)

    def backward(self, dout):
        eps = self.eps
        dout = dout.ravel().reshape(dout.shape[0], -1)

        X_mu = self.X_flat - self.mu
        var_inv = 1. / np.sqrt(self.var + eps)

        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(dout * self.X_hat, axis=0)

        dX_hat = dout * self.gamma
        dvar = np.sum(dX_hat * X_mu, axis=0) * -0.5 * (self.var + eps) ** (-3 / 2)
        dmu = np.sum(dX_hat * (-var_inv), axis=0) + dvar * 1 / self.n_X * np.sum(-2. * X_mu, axis=0)
        dX = (dX_hat * var_inv) + (dmu / self.n_X) + (dvar * 2 / self.n_X * X_mu)
        dX = dX.reshape(self.X_shape)

        self.grads[0] += dgamma
        self.grads[1] += dbeta
        return dX  # , dgamma, dbeta


# =============================================================================
# BatchNorm：可處理 4D 輸入（N, C, H, W）的 BatchNorm 版本
# =============================================================================
class BatchNorm(Layer):
    def __init__(self, num_features, gamma_beta_method=None, eps=1e-6, momentum=0.9, std=0.02):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        if not gamma_beta_method:
            self.gamma = np.ones((1, num_features))
            self.beta = np.zeros((1, num_features))
        else:
            self.gamma = np.random.normal(1, std, (1, num_features))
            self.beta = np.zeros((1, num_features))

        self.params = [self.gamma, self.beta]
        self.grads = [np.zeros_like(self.gamma), np.zeros_like(self.beta)]

        self.running_mu = np.zeros((1, num_features))
        self.running_var = np.zeros((1, num_features))

    def forward(self, X, training=True):
        N, C, H, W = X.shape
        self.X_shape = X.shape

        if training:
            # 若是 4D：把通道軸移到最後，方便 reshape 成 (-1, C)
            if len(self.X_shape) > 2:
                X = np.moveaxis(X, 1, 3)
                X_flat = X.reshape(-1, X.shape[3])
            else:
                X_flat = X

            NHW = X_flat.shape[0]
            self.n_X = NHW

            mu = np.mean(X_flat, axis=0)
            var = 1 / float(NHW) * np.sum((X_flat - mu) ** 2, axis=0)
            X_hat = (X_flat - mu) / np.sqrt(var + self.eps)

            out = self.gamma * X_hat + self.beta

            if len(self.X_shape) > 2:
                out = out.reshape(N, H, W, C)
                out = np.moveaxis(out, 3, 1)

            self.mu, self.var, self.X_flat, self.X_hat = mu, var, X_flat, X_hat

            # 更新 running mean/var（原碼用區域變數接住，但不回寫 self.running_*）
            running_mu, running_var, momentum = self.running_mu, self.running_var, self.momentum
            running_mu = momentum * running_mu + (1 - momentum) * self.mu
            running_var = momentum * running_var + (1 - momentum) * self.var

        else:
            if len(self.X_shape) > 2:
                X = np.moveaxis(X, 1, 3)
                self.X_flat = X.reshape(-1, X.shape[3])
            else:
                self.X_flat = X

            # 正規化（注意：原碼用 eps 變數，但此分支沒有定義 eps；依要求不修正）
            X_hat = (X_flat - self.running_mu) / np.sqrt(self.running_var + eps)
            out = self.gamma * X_hat + self.beta

            if len(self.X_shape) > 2:
                out = out.reshape(N, H, W, C)
                out = np.moveaxis(out, 3, 1)

        return out

    def __call__(self, X):
        return self.forward(X)

    def backward(self, dout):
        # 若輸出是 4D，先把 C 軸移到最後再 reshape 成 (-1, C)
        if len(dout.shape) > 2:
            dout = np.moveaxis(dout, 1, 3)
            dout = dout.reshape(-1, dout.shape[3])

        N, D = dout.shape
        assert (N == self.n_X)

        mu, var, x, x_hat, eps = self.mu, self.var, self.X_flat, self.X_hat, self.eps
        gamma, beta = self.gamma, self.beta

        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(dout * x_hat, axis=0)

        X_mu = x - mu
        var_inv = 1. / np.sqrt(var + eps)

        if True:
            dX_hat = dout * gamma
            dvar = np.sum(dX_hat * X_mu, axis=0) * (-0.5) * (var + eps) ** (-3 / 2)
            dmu = np.sum(dX_hat * (-var_inv), axis=0) + dvar * 1 / N * np.sum(-2. * X_mu, axis=0)
            dX = (dX_hat * var_inv) + (dmu / N) + (dvar * 2 / N * X_mu)
        else:
            # 原碼保留的另一種寫法
            x_mu = x - mu
            dx_hat = dout * gamma
            dxmu1 = dx_hat * 1 / np.sqrt(var + eps)
            divar = np.sum(dx_hat * x_mu, axis=0)
            dvar = divar * -1 / 2 * (var + eps) ** (-3 / 2)
            dsq = 1 / N * np.ones((N, D)) * dvar
            dxmu2 = 2 * x_mu * dsq
            dx1 = dxmu1 + dxmu2
            dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)
            dx2 = 1 / N * np.ones((N, D)) * dmu
            dX = dx1 + dx2

        if len(self.X_shape) > 2:
            N, C, H, W = self.X_shape
            dX = dX.reshape(N, H, W, C)
            dX = np.moveaxis(dX, 3, 1)

        self.grads[0] += dgamma
        self.grads[1] += dbeta
        return dX  # , dgamma, dbeta


# =============================================================================
# Dropout
# =============================================================================
# https://deepnotes.io/dropout
class Dropout(Layer):
    def __init__(self, dropout_p=0.5, seed=None):
        super().__init__()
        # retain_p：保留機率（inverted dropout 會用到）
        self.retain_p = 1 - dropout_p
        self._mask = None
        if seed:
            np.random.seed(seed)

    def forward(self, x, training=True):
        retain_p = self.retain_p
        if training:
            # inverted dropout：mask / retain_p，維持期望值一致
            self._mask = (np.random.rand(*x.shape) < retain_p) / retain_p
            out = x * self._mask
        else:
            out = x
        return out

    def __call__(self, x, training=True):
        return self.forward(x, training)

    def backward(self, grad_output, training=True):
        if training:
            dx = grad_output * self._mask
        else:
            dx = grad_output
        return dx


# =============================================================================
# Conv：朴素版卷積（for 迴圈版）
# =============================================================================
class Conv(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.C = in_channels
        self.F = out_channels
        self.K = kernel_size
        self.S = stride
        self.P = padding

        # 權重： (F, C, K, K)
        self.W = np.random.randn(self.F, self.C, self.K, self.K)  # /(self.K*self.K)
        # 偏置： (out_channels,)
        self.b = np.random.randn(out_channels,)

        self.params = [self.W, self.b]
        self.grads = [np.zeros_like(self.W), np.zeros_like(self.b)]
        self.X = None

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform(self.W, a=math.sqrt(5))
        if self.b is not None:
            # fan_in = self.C（原碼註解保留）
            fan_in = self.C
            bound = 1 / math.sqrt(fan_in)
            self.b[:] = np.random.uniform(-bound, bound, (self.b.shape))

    def forward(self, X):
        self.X = X
        N, C, X_h, X_w = self.X.shape
        F, _, F_h, F_w = self.W.shape

        # padding：補零
        X_pad = np.pad(
            self.X,
            ((0, 0), (0, 0), (self.P, self.P), (self.P, self.P)),
            mode='constant',
            constant_values=0
        )

        # 輸出大小
        O_h = 1 + int((X_h + 2 * self.P - F_h) / self.S)
        O_w = 1 + int((X_w + 2 * self.P - F_w) / self.S)
        O = np.zeros((N, F, O_h, O_w))

        # 朴素卷積：逐點計算
        for n in range(N):
            for f in range(F):
                for i in range(O_h):
                    hs = i * self.S
                    for j in range(O_w):
                        ws = j * self.S
                        O[n, f, i, j] = (X_pad[n, :, hs:hs + F_h, ws:ws + F_w] * self.W[f]).sum() + self.b[f]

        return O

    def __call__(self, X):
        return self.forward(X)

    def backward(self, dZ):
        """
        朴素卷積 backward（逐點累積 dX、dW、db）
        """
        N, F, Z_h, Z_w = dZ.shape
        N, C, X_h, X_w = self.X.shape
        F, _, F_h, F_w = self.W.shape

        pad = self.P

        H_ = 1 + (X_h + 2 * pad - F_h) // self.S
        W_ = 1 + (X_w + 2 * pad - F_w) // self.S

        dX = np.zeros_like(self.X)
        dW = np.zeros_like(self.W)
        db = np.zeros_like(self.b)

        X_pad = np.pad(self.X, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
        dX_pad = np.pad(dX, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')

        for n in range(N):
            for f in range(F):
                db[f] += dZ[n, f].sum()
                for i in range(H_):
                    hs = i * self.S
                    for j in range(W_):
                        ws = j * self.S

                        # dW：輸入區塊 * 上游梯度
                        dW[f] += X_pad[n, :, hs:hs + F_h, ws:ws + F_w] * dZ[n, f, i, j]
                        # dX_pad：卷積核 * 上游梯度
                        dX_pad[n, :, hs:hs + F_h, ws:ws + F_w] += self.W[f] * dZ[n, f, i, j]

        # 去掉 padding（原碼切片保留）
        dX = dX_pad[:, :, pad:pad + X_h, pad:pad + X_h]

        self.grads[0] += dW
        self.grads[1] += db

        return dX

    # -------- 添加正則項的梯度 --------
    def reg_grad(self, reg):
        self.grads[0] += 2 * reg * self.W

    def reg_loss(self, reg):
        return reg * np.sum(self.W ** 2)

    def reg_loss_grad(self, reg):
        self.grads[0] += 2 * reg * self.W
        return reg * np.sum(self.W ** 2)


# =============================================================================
# Pool：Max Pooling
# =============================================================================
class Pool(Layer):
    def __init__(self, pool_param=(2, 2, 2)):
        super().__init__()
        self.pool_h, self.pool_w, self.stride = pool_param

    def forward(self, x):
        self.x = x
        N, C, H, W = x.shape

        pool_h, pool_w, stride = self.pool_h, self.pool_w, self.stride

        h_out = 1 + (H - pool_h) // stride
        w_out = 1 + (W - pool_w) // stride
        out = np.zeros((N, C, h_out, w_out))

        for n in range(N):
            for c in range(C):
                for i in range(h_out):
                    si = stride * i
                    for j in range(w_out):
                        sj = stride * j
                        x_win = x[n, c, si:si + pool_h, sj:sj + pool_w]
                        out[n, c, i, j] = np.max(x_win)

        return out

    def backward(self, dout):
        x = self.x
        N, C, H, W = x.shape
        kH, kW, stride = self.pool_h, self.pool_w, self.stride

        oH = 1 + (H - kH) // stride
        oW = 1 + (W - kW) // stride

        dx = np.zeros_like(x)

        for k in range(N):
            for l in range(C):
                for i in range(oH):
                    si = stride * i
                    for j in range(oW):
                        sj = stride * j
                        slice = x[k, l, si:si + kH, sj:sj + kW]
                        slice_max = np.max(slice)
                        dx[k, l, si:si + kH, sj:sj + kW] += (slice_max == slice) * dout[k, l, i, j]

        return dx


# =============================================================================
# BatchNorm2D：針對 NCHW 的 BatchNorm（以通道為特徵）
# =============================================================================
class BatchNorm2D(Layer):
    def __init__(self, num_features, gamma_beta_method=None, eps=1e-6, momentum=0.9, std=0.02):
        super().__init__()
        self.eps = eps
        self.momentum = momentum

        if not gamma_beta_method:
            self.gamma = np.ones((1, num_features, 1, 1))
            self.beta = np.zeros((1, num_features, 1, 1))
        else:
            self.gamma = np.random.normal(1, std, (1, num_features, 1, 1))
            self.beta = np.zeros((1, num_features, 1, 1))

        self.params = [self.gamma, self.beta]
        self.grads = [np.zeros_like(self.gamma), np.zeros_like(self.beta)]

        self.running_mu = np.zeros((1, num_features, 1, 1))
        self.running_var = np.zeros((1, num_features, 1, 1))

    def forward(self, x, training=True):
        eps = self.eps
        running_mu, running_var, momentum = self.running_mu, self.running_var, self.momentum
        gamma, beta = self.gamma, self.beta

        N, C, H, W = x.shape

        if training:
            mu = np.mean(x, axis=(0, 2, 3)).reshape(1, C, 1, 1)
            var = 1 / float(N * H * W) * np.sum((x - mu) ** 2, axis=(0, 2, 3)).reshape(1, C, 1, 1)
            x_hat = (x - mu) / np.sqrt(var + eps)
            y = gamma.reshape(1, C, 1, 1) * x_hat + beta.reshape(1, C, 1, 1)
            out = y

            running_mu = momentum * running_mu + (1 - momentum) * mu
            running_var = momentum * running_var + (1 - momentum) * var

            self.x_hat, self.mu, self.var, self.x = x_hat, mu, var, x

        else:
            x_hat = (x - running_mu) / np.sqrt(running_var + eps)
            y = gamma.reshape(1, C, 1, 1) * x_hat + beta.reshape(1, C, 1, 1)
            out = y

        return out

    def __call__(self, X):
        return self.forward(X)

    def backward(self, dout):
        x_hat, mu, var, eps, gamma, beta, x = self.x_hat, self.mu, self.var, self.eps, self.gamma, self.beta, self.x
        N, C, H, W = dout.shape

        dbeta = np.sum(dout, axis=(0, 2, 3))
        dgamma = np.sum(dout * x_hat, axis=(0, 2, 3))
        dgamma = dgamma.reshape(1, C, 1, 1)
        dbeta = dbeta.reshape(1, C, 1, 1)

        # 反向推導參考（原碼保留）
        # https://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html

        gamma_reshape = gamma.reshape(1, C, 1, 1)
        beta_reshape = beta.reshape(1, C, 1, 1)
        Nt = N * H * W

        dx_hat = dout * gamma_reshape
        dxmu1 = dx_hat * 1 / np.sqrt(var + eps)
        divar = np.sum(dx_hat * (x - mu), axis=(0, 2, 3)).reshape(1, C, 1, 1)
        dvar = divar * -1 / 2 * (var + eps) ** (-3 / 2)
        dsq = 1 / Nt * np.broadcast_to(np.broadcast_to(np.squeeze(dvar), (W, H, C)).transpose(2, 1, 0), (N, C, H, W))
        dxmu2 = 2 * (x - mu) * dsq
        dx1 = dxmu1 + dxmu2
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=(0, 2, 3))
        dx2 = 1 / Nt * np.broadcast_to(np.broadcast_to(np.squeeze(dmu), (W, H, C)).transpose(2, 1, 0), (N, C, H, W))
        dx = dx1 + dx2

        # 原碼 debug print 保留
        print("dgamma.shape", dgamma.shape)

        self.grads[0] += dgamma
        self.grads[1] += dbeta
        return dx


# =============================================================================
# Conv_fast：im2row/row2im 加速卷積（矩陣乘法版）
# =============================================================================
class Conv_fast():
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.C = in_channels
        self.F = out_channels
        self.kH = kernel_size
        self.kW = kernel_size
        self.S = stride
        self.P = padding

        self.K = np.random.normal(0, 1, (self.F, self.C, self.kH, self.kW))
        self.b = np.zeros((1, self.F))

        self.params = [self.K, self.b]
        self.grads = [np.zeros_like(self.K), np.zeros_like(self.b)]
        self.X = None

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform(self.K, a=math.sqrt(5))
        if self.b is not None:
            fan_in = self.C
            bound = 1 / math.sqrt(fan_in)
            self.b[:] = np.random.uniform(-bound, bound, (self.b.shape))

    def forward(self, X):
        # 轉成多通道形式
        self.X = X
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1, 1, 1)
        elif len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1, 1)

        self.N, self.H, self.W = X.shape[0], X.shape[2], X.shape[3]
        S, P, kH, kW = self.S, self.P, self.kH, self.kW
        self.oH = (self.H - kH + 2 * P) // S + 1
        self.oW = (self.W - kW + 2 * P) // S + 1

        X_shape = (self.N, self.C, self.H, self.W)

        if True:
            self.X_row = im2row_indices(X, self.kH, self.kW, S=self.S, P=self.P)
        else:
            if P == 0:
                X_pad = X
            else:
                X_pad = np.pad(X, ((0, 0), (0, 0), (P, P), (P, P)), 'constant')
            self.X_row = im2row(X_pad, kH, kW, S)

        K_col = self.K.reshape(self.F, -1).transpose()
        Z_row = self.X_row @ K_col + self.b

        Z = Z_row.reshape(self.N, self.oH, self.oW, -1)
        Z = Z.transpose(0, 3, 1, 2)
        return Z

    def __call__(self, x):
        return self.forward(x)

    def backward(self, dZ):
        if len(dZ.shape) <= 2:
            dZ = dZ.reshape(dZ.shape[0], -1, self.oH, self.oW)

        K = self.K

        # 將 dZ 轉成 dZ_row： (N*oH*oW, F)
        F = dZ.shape[1]
        assert (F == self.F)
        dZ_row = dZ.transpose(0, 2, 3, 1).reshape(-1, F)

        # dK
        dK_col = np.dot(self.X_row.T, dZ_row)
        dK_col = dK_col.transpose(1, 0)
        dK = dK_col.reshape(self.K.shape)

        # db
        db = np.sum(dZ, axis=(0, 2, 3))
        db = db.reshape(-1, F)

        # dX_row
        K_col = K.reshape(K.shape[0], -1).transpose()
        dX_row = np.dot(dZ_row, K_col.T)

        if True:
            X_shape = (self.N, self.C, self.H, self.W)
            dX = row2im_indices(dX_row, X_shape, self.kH, self.kW, S=self.S, P=self.P)
        else:
            dX_pad = row2im(dX_row, oH, oW, kH, kW, S)
            if P == 0:
                dX = dX_pad
            dX = dX_pad[:, :, P:-P, P:-P]

        dX = dX.reshape(self.X.shape)

        self.grads[0] += dK
        self.grads[1] += db

        return dX

    # -------- 添加正則項的梯度 --------
    def reg_grad(self, reg):
        self.grads[0] += 2 * reg * self.K

    def reg_loss(self, reg):
        return reg * np.sum(self.K ** 2)

    def reg_loss_grad(self, reg):
        self.grads[0] += 2 * reg * self.K
        return reg * np.sum(self.K ** 2)


# =============================================================================
# Conv_transpose：轉置卷積（反卷積）
# =============================================================================
class Conv_transpose():
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.C = in_channels
        self.F = out_channels
        self.kH = kernel_size
        self.kW = kernel_size
        self.S = stride
        self.P = padding

        self.K = np.random.normal(0, 1, (self.C, self.F, self.kH, self.kW))
        self.b = np.zeros((1, self.F))

        self.params = [self.K, self.b]
        self.grads = [np.zeros_like(self.K), np.zeros_like(self.b)]
        self.X = None

        self.reset_parameters()

    def reset_parameters(self):
        kaiming_uniform(self.K, a=math.sqrt(5))
        if self.b is not None:
            fan_in, _ = calculate_fan_in_and_fan_out(self.K)
            bound = 1 / math.sqrt(fan_in)
            self.b[:] = np.random.uniform(-bound, bound, (self.b.shape))

    def forward(self, X):
        '''
        X:       (N,C,H,W)
        K:       (F,C,kH,kW)  # 注意：此處註解與實際 K shape 可能不一致，依原碼保留
        Z:       (N,F,oH,oW)
        X_row:   (N*oH*oW, C*kH*kW)
        K_col:    (C*kH*kW, F)
        Z_row = X_row*K_col:  (N*oH*oW, C*kH*kW)*(C*kH*kW, F) =  (N*oH*oW, F)

        dK_col = X_row.T @dZ_row: (C*kH*kW,N*oH*oW)*(N*oH*oW, F) = (C*kH*kW,F)
        dX_row = dZ_row@K_col.T = (N*oH*oW, F) * (F, C*kH*kW) = (N*oH*oW, C*kH*kW)
        '''
        self.X = X

        # 轉成多通道
        if len(X.shape) == 1:
            X = X.reshape(X.shape[0], 1, 1, 1)
        elif len(X.shape) == 2:
            X = X.reshape(X.shape[0], X.shape[1], 1, 1)

        self.N, self.H, self.W = X.shape[0], X.shape[2], X.shape[3]
        S, P, kH, kW = self.S, self.P, self.kH, self.kW
        self.oH = self.S * (self.H - 1) + kH - 2 * P
        self.oW = self.S * (self.W - 1) + kW - 2 * P

        K = self.K
        F = X.shape[1]

        X_row = X.transpose(0, 2, 3, 1).reshape(-1, F)
        K_col = K.reshape(K.shape[0], -1).transpose()

        Z_row = np.dot(X_row, K_col.T)

        Z_shape = (self.N, self.F, self.oH, self.oW)
        Z = row2im_indices(Z_row, Z_shape, self.kH, self.kW, S=self.S, P=self.P)

        self.b = self.b.reshape(1, self.F, 1, 1)
        Z += self.b

        self.X_row = X_row
        return Z

    def __call__(self, X):
        return self.forward(X)

    def backward(self, dZ):
        N, F, oH, oW = dZ.shape[0], dZ.shape[1], dZ.shape[2], dZ.shape[3]
        S, P, kH, kW = self.S, self.P, self.kH, self.kW

        dZ_row = im2row_indices(dZ, self.kH, self.kW, S=self.S, P=self.P)

        K_col = self.K.reshape(self.K.shape[0], -1).transpose()

        dX_row = dZ_row @ K_col
        dK_col = self.X_row.T @ dZ_row

        dK = dK_col.reshape(self.K.shape)

        db = np.sum(dZ, axis=(0, 2, 3))
        db = db.reshape(-1, F)

        dX = dX_row.reshape(N, self.H, self.W, self.C)
        dX = dX.transpose(0, 3, 1, 2)

        self.grads[0] += dK
        self.grads[1] += db

        return dX

    # -------- 添加正則項的梯度 --------
    def reg_grad(self, reg):
        self.grads[0] += 2 * reg * self.K

    def reg_loss(self, reg):
        return reg * np.sum(self.K ** 2)

    def reg_loss_grad(self, reg):
        self.grads[0] += 2 * reg * self.K
        return reg * np.sum(self.K ** 2)


# =============================================================================
# one-hot / 向量化工具
# 注意：以下開始原碼有「重複定義」與「重複 import numpy」，
#       依你的要求不刪、不改，只加註解提醒。
# =============================================================================
# https://pytorch.org/docs/master/_modules/torch/nn/modules/sparse.html#Embedding

def one_hot(size, idx_seq):
    x = []
    for idx in idx_seq:
        v = [0 for _ in range(size)]
        v[idx] = 1
        x.append(v)
    return x

def vecterization(size, indexes):
    # dtype=np.float 可能在新版 numpy 會警告，但依要求不修正
    x = np.array(one_hot(size, indexes), dtype=np.float)  # (seq_len, input_size)
    x = np.expand_dims(x, axis=1)  # batch_size=1：(seq_len, 1, input_size)
    return x


import numpy as np  # ※ 重複 import（保留原碼行為）
def one_hot(size, indices, expend=False):
    """
    ※ 注意：這裡的 one_hot 會覆蓋前面同名 one_hot（Python 後定義者生效）
    """
    x = np.eye(size)[indices.reshape(-1)]
    if expend:
        x = np.expand_dims(x, axis=1)
    return x

def one_got_np(size, indices, expend=False):
    x = np.array(one_hot(size, indices), dtype=np.float)
    if expend:
        x = np.expand_dims(x, axis=1)
    return x

# one_hot(10,[3,2,5])
# vecterization(10,[3,2,5])


# =============================================================================
# Embedding：查表式嵌入層
# =============================================================================
class Embedding():
    def __init__(self, num_embeddings, embedding_dim, _weight=None):
        super().__init__()

        if _weight is None:
            self.W = np.empty((num_embeddings, embedding_dim))
            self.reset_parameters()
            self.preTrained = False
        else:
            self.W = _weight
            self.preTrained = True

        self.params = [self.W]
        self.grads = [np.zeros_like(self.W)]

    def reset_parameters(self):
        self.W[:] = np.random.randn(*self.W.shape)

    def forward(self, indices):
        num_embeddings = self.W.shape[0]

        # 這裡建立 one-hot 主要是給 backward 算 dW 用
        x = one_hot(num_embeddings, indices).astype(float)
        self.x = x

        # 直接查表（advanced indexing）
        Z = self.W[indices, :]
        return Z

    def __call__(self, indices):
        return self.forward(indices)

    def backward(self, dZ):
        x = self.x
        dW = np.dot(x.T, dZ)
        dx = np.dot(dZ, np.transpose(self.W))

        self.grads[0] += dW
        return dx


# =============================================================================
# softmax / attention 相關函式
# =============================================================================
# import util

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

def softmax_backward_2(Z, dF, isF=True):
    """
    softmax 的反向（使用 Jacobian）
    - isF=True：Z 就是 softmax 後的機率 F
    - isF=False：Z 是 logits，需要先 softmax(Z)
    """
    if isF:
        F = Z
    else:
        F = softmax(Z)

    D = []
    for i in range(F.shape[0]):
        f = F[i]
        D.append(np.diag(f.flatten()))

    grads = D - np.einsum('ij,ik->ijk', F, F)
    grad = np.einsum("bj, bjk -> bk", dF, grads)  # [B,D]*[B,D,D] -> [B,D]
    return grad


def attn_forward(hidden, encoder_outputs):
    # hidden (B,D)   encoder_outputs (T,B,D)
    energies = np.sum(hidden * encoder_outputs, axis=2)  # (T,B)
    energies = energies.T  # (B,T)
    alphas = util.softmax(energies)
    return alphas, energies


def attn_backward(d_alpha, energies, hidden, encoder_outputs):
    # hidden (B,D)   encoder_outputs (T,B,D)
    # d_alpha energies:(B,T)
    d_energies = softmax_backward_2(energies, d_alpha, False)
    d_energies = d_energies.T  # (T,B)
    d_energies = np.expand_dims(d_energies, axis=2)

    d_encoder_outputs = d_energies * hidden  # (T,B,1)*(B,D) -> (T,B,D)
    d_hidden = np.sum(d_energies * encoder_outputs, axis=0)  # (T,B,D) -> (B,D)
    return d_encoder_outputs, d_hidden


def bmm(alphas, encoder_outputs):
    # (B,T), (T,B,D)
    encoder_outputs = np.transpose(encoder_outputs, (1, 0, 2))  # [T,B,D] -> [B,T,D]
    context = np.einsum("bj, bjk -> bk", alphas, encoder_outputs)  # [B,T]*[B,T,D] -> [B,D]
    return context

def bmm_backward(d_context, alphas, encoder_outputs):
    encoder_outputs = np.transpose(encoder_outputs, (1, 0, 2))  # [T,B,D] -> [B,T,D]
    d_alphas = np.einsum("bjk, bk -> bj", encoder_outputs, d_context)  # (B,T)
    d_encoder_outputs = np.einsum("bi, bj -> bij", alphas, d_context)  # (B,T,D)
    d_encoder_outputs = np.transpose(d_encoder_outputs, (1, 0, 2))  # [B,T,D] -> [T,B,D]
    return d_alphas, d_encoder_outputs


# =============================================================================
# Atten：注意力層（封裝 attn_forward + bmm）
# =============================================================================
# Attention layer at a time t
class Atten(Layer):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, hidden, encoder_outputs):
        self.hidden = hidden
        self.encoder_outputs = encoder_outputs

        alphas, energies = attn_forward(hidden, encoder_outputs)
        context = bmm(alphas, encoder_outputs)

        self.alphas, self.energies = alphas, energies
        return context, alphas, energies

    def __call__(self, hidden, encoder_outputs):
        return self.forward(hidden, encoder_outputs)

    def backward(self, d_context):  # (B,D)
        alphas, energies, hidden, encoder_outputs = self.alphas, self.energies, self.hidden, self.encoder_outputs

        d_alphas, d_encoder_outputs_2 = bmm_backward(d_context, alphas, encoder_outputs)
        d_encoder_outputs, d_hidden = attn_backward(d_alphas, energies, hidden, encoder_outputs)

        d_encoder_outputs += d_encoder_outputs_2
        return d_hidden, d_encoder_outputs
