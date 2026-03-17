import numpy as np
import math

np.random.seed(1)

def rnn_params_init(input_dim, hidden_dim, output_dim, scale = 0.01):        
    # Input -> Hidden 的權重
    Wx = np.random.randn(input_dim, hidden_dim) * scale 
    
    # Hidden -> Hidden 的遞迴權重
    Wh = np.random.randn(hidden_dim, hidden_dim) * scale 
    
    # Hidden 層偏置
    bh = np.zeros((1, hidden_dim)) 

    # Hidden -> Output 的權重
    Wf = np.random.randn(hidden_dim, output_dim) * scale 
    
    # Output 層偏置
    bf = np.zeros((1, output_dim)) 

    # 回傳所有參數
    return [Wx, Wh, bh, Wf, bf]


def rnn_hidden_state_init(batch_dim, hidden_dim):
    # 初始化 hidden state
    return np.zeros((batch_dim, hidden_dim))

def rnn_forward(params, Xs, H_):
    # 拆出參數
    Wx, Wh, bh, Wf, bf = params
    
    # 目前的 hidden state
    H = H_ #np.copy(H_)   
   
    Fs = []        # 存每個時間點的輸出
    Hs = {}        # 存每個時間點的 hidden state
    
    # 先存初始 hidden state
    Hs[-1] = np.copy(H)    
 
    # 依序跑每個時間點
    for t in range(len(Xs)):
        X = Xs[t]       # 目前時間點的輸入
        
        # 更新 hidden state
        H = np.tanh(np.dot(X, Wx) + np.dot(H, Wh) + bh)
        
        # 算 output
        F = np.dot(H, Wf) + bf       

        # 存結果
        Fs.append(F)
        Hs[t] = H
        
    # 回傳所有時間點的輸出與 hidden state
    return Fs, Hs

def softmax(Z):
    # 先減最大值，避免數值太大
    A = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
    # 正規化成機率分布
    return A / np.sum(A, axis=-1, keepdims=True)

def softmax_cross_entropy(Z, y, onehot=False):
    # 樣本數
    m = len(Z)

    # 先做 softmax
    F = softmax(Z)

    if onehot:
        # y 是 one-hot 時的 loss
        loss = -np.sum(y * np.log(F)) / m
    else:
        # y 是類別 index 時
        y.flatten()

        # 取正確類別的 log 機率
        log_Fy = -np.log(F[range(m), y])

        # 平均 loss
        loss = np.sum(log_Fy) / m

    return loss

def cross_entropy_grad_loss(Z, y, softmax_out=False, onehot=False):
    """
    一次回傳 cross entropy loss 跟 gradient
    """
    # 決定要不要先做 softmax
    if softmax_out:
        F = Z
    else:
        F = softmax(Z)

    # 計算 loss
    m = len(y)
    if onehot:
        loss = -np.sum(y * np.log(F + 1e-12)) / m
    else:
        loss = -np.sum(np.log(F[np.arange(m), y] + 1e-12)) / m

    # 計算 gradient
    grad = F.copy()
    if onehot:
        grad = (grad - y) / m
    else:
        grad[np.arange(m), y] -= 1
        grad /= m

    return loss, grad

def rnn_loss_grad(Fs, Ys, loss_fn = cross_entropy_grad_loss, flatten = True):   
    # 初始化總 loss
    loss = 0
    
    # 存每個時間點的輸出梯度
    dFs = {}
   
    # 逐時間點計算 loss
    for t in range(len(Fs)):
        F = Fs[t]      # 預測輸出
        Y = Ys[t]      # 正確答案
        
        # 視情況攤平成一維
        if flatten and Y.ndim >= 2:          
            Y = Y.flatten()
            
        # 算該時間點的 loss 跟梯度
        loss_t, dF_t = loss_fn(F, Y)
        
        # 累加 loss
        loss += loss_t        
        
        # 存梯度
        dFs[t] = dF_t
       
    # 回傳總 loss 跟各時間點梯度
    return loss, dFs

# 梯度裁切，避免梯度爆掉
def grad_clipping(grads, alpha):
    # 算所有梯度的 L2 norm
    norm = math.sqrt(sum((grad ** 2).sum() for grad in grads))
    # 超過門檻就縮小
    if norm > alpha:
        ratio = alpha / norm
        for i in range(len(grads)):
            grads[i] *= ratio 

def rnn_loss_grad(Fs, Ys, loss_fn = cross_entropy_grad_loss, flatten = True):   
    # 初始化總 loss
    loss = 0
    
    # 存每個時間點的輸出梯度
    dFs = {}
   
    # 逐時間點計算 loss
    for t in range(len(Fs)):
        F = Fs[t]      # 預測輸出
        Y = Ys[t]      # 正確答案
        
        # 視情況攤平成一維
        if flatten and Y.ndim >= 2:          
            Y = Y.flatten()
            
        # 算該時間點的 loss 跟梯度
        loss_t, dF_t = loss_fn(F, Y)
        
        # 累加 loss
        loss += loss_t        
        
        # 存梯度
        dFs[t] = dF_t
       
    # 回傳總 loss 跟各時間點梯度
    return loss, dFs
            
def rnn_backward(params, Xs, Hs, dZs, clip_value = 5.):    

    # 拆出參數
    Wx, Wh, bh, Wf, bf = params
    
    # 初始化各參數的梯度
    dWx, dWh, dWf = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(Wf)
    dbh, dbf = np.zeros_like(bh), np.zeros_like(bf)       

    # 來自下一個時間點的 hidden 梯度
    dh_next = np.zeros_like(Hs[0])
    h = Hs
    x = Xs
    
    T = len(Xs)  # 時間步數
    
    # 反向跑時間序列
    for t in reversed(range(T)): 
        dZ = dZs[t]  # 該時間點 output 的梯度
        
        # Output 層梯度
        dWf += np.dot(h[t].T, dZ)
        dbf += np.sum(dZ, axis=0, keepdims=True)         
        
        # 傳回 hidden 層的梯度
        dh = np.dot(dZ, Wf.T) + dh_next 
        
        # tanh 的微分
        dZh = (1 - h[t] * h[t]) * dh 
        
        # Hidden 層梯度
        dbh += np.sum(dZh, axis=0, keepdims=True) 
        dWx += np.dot(x[t].T, dZh)
        dWh += np.dot(h[t-1].T, dZh)
        
        # 繼續往前一個時間點傳
        dh_next = np.dot(dZh, Wh.T)
   
    # 整理成梯度列表
    grads = [dWx, dWh, dbh, dWf, dbf]
    
    # 需要的話做梯度裁切
    if clip_value is not None:
        grad_clipping(grads, clip_value)
        
    return grads

class SGD():
    # 最基本的 SGD + Momentum optimizer

    def __init__(self, model_params, learning_rate=0.01, momentum=0.9):
        # 存參數與超參數
        self.params = model_params
        self.lr = learning_rate
        self.momentum = momentum
        
        # 初始化每個參數對應的速度
        self.vs = []
        for p in self.params:
            v = np.zeros_like(p)
            self.vs.append(v)   
                

    def step(self, grads): 
        # 更新所有參數
        for i in range(len(self.params)):   
            grad = grads[i]  # 目前參數的梯度
            
            # 更新速度
            self.vs[i] = self.momentum * self.vs[i] + self.lr * grad
            
            # 更新參數
            self.params[i] -= self.vs[i]


    def scale_learning_rate(self, scale):
        # 縮放 learning rate
        self.lr *= scale

class AdaGrad():
    # AdaGrad optimizer

    def __init__(self, model_params, learning_rate=0.01):
        # 存參數與 learning rate
        self.params = model_params
        self.lr = learning_rate
        
        # 累積梯度平方和
        self.vs = []
        
        # 避免除以 0
        self.delta = 1e-7
        
        for p in self.params:
            v = np.zeros_like(p)
            self.vs.append(v)       
                 
    def step(self, grads): 
        # 更新所有參數
        for i in range(len(self.params)):  
            grad = grads[i]
            
            # 累積梯度平方
            self.vs[i] += grad**2
            
            # 依照 AdaGrad 公式更新參數
            self.params[i] -= self.lr * grad / (self.delta + np.sqrt(self.vs[i]))
            
    def scale_learning_rate(self, scale):
        # 縮放 learning rate
        self.lr *= scale

def mse_loss_grad(f, y):
    # 樣本數
    m = len(f)

    # 計算 MSE loss
    loss = (1./m) * np.sum((f - y) ** 2)

    # 計算對 f 的梯度
    grad = (2./m) * (f - y)

    # 回傳 loss 跟梯度
    return loss, grad