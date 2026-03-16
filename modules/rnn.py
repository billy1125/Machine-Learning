import numpy as np
import math

np.random.seed(1)

def rnn_params_init(input_dim, hidden_dim, output_dim, scale = 0.01):        
    # Input → Hidden 權重
    Wx = np.random.randn(input_dim, hidden_dim) * scale 
    
    # Hidden → Hidden 權重（時間上的遞迴連結）
    Wh = np.random.randn(hidden_dim, hidden_dim) * scale 
    
    # Hidden 層偏置
    bh = np.zeros((1, hidden_dim)) 

    # Hidden → Output 權重
    Wf = np.random.randn(hidden_dim, output_dim) * scale 
    
    # Output 層偏置
    bf = np.zeros((1, output_dim)) 

    # 回傳所有參數
    return [Wx, Wh, bh, Wf, bf]


def rnn_hidden_state_init(batch_dim, hidden_dim):
    # 初始化 hidden state（通常一開始設為 0）
    return np.zeros((batch_dim, hidden_dim))

def rnn_forward(params, Xs, H_):
    # 參數
    Wx, Wh, bh, Wf, bf = params
    
    # 初始的隱藏狀態 (Hidden State) 
    H = H_ #np.copy(H_)   
   
    Fs = []        # 用來存每一格產出的 Output
    Hs = {}        # 用來存每一格產出的 Hidden State，方便之後做 Backprop
    
    # 把「前一時刻」的狀態（也就是 -1）先存進去
    Hs[-1] = np.copy(H)    
 
    # 開始跑 Time Steps 的迴圈
    for t in range(len(Xs)):
        X = Xs[t]       # 抓出當前時間點的輸入資料
        
        # 這是 RNN 的心臟：把目前的輸入、前一格的狀態加權之後，丟進 tanh 擠壓一下
        # 這裡的 np.dot(X, Wx) 是把目前的輸入 X 跟 Input 到 Hidden 層的權重 Wx 做矩陣乘法，算出對 Hidden 層的貢獻
        H = np.tanh(np.dot(X, Wx) + np.dot(H, Wh) + bh)
        
        # 根據算好的 H，再過一層線性轉換算出最後的輸出結果 F
        # 這裡的 np.dot(H, Wf) 是把目前的 Hidden State H 跟 Hidden 到 Output 層的權重 Wf 做矩陣乘法，算出對 Output 層的貢獻
        # 這裡的 H 的形狀是 (1, 10)，Wf 的形狀是 (10, 4)，bf 的形狀是 (1, 4)，所以 np.dot(H, Wf) 的結果是 (1, 4)，也就是對 Output 層的貢獻
        F = np.dot(H, Wf) + bf       

        # 把這一步算出來的結果通通塞進 list 跟 dict 裡面
        Fs.append(F)
        Hs[t] = H
        
    # 最後把這一整串算出來的 Fs 跟 Hs 吐回去
    return Fs, Hs

def softmax(Z):
    # 對輸入的 Z 做 softmax，先減掉最大值避免數值爆掉
    A = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
    # 將指數結果做正規化，讓每一列加起來等於 1
    return A / np.sum(A, axis=-1, keepdims=True)

def softmax_cross_entropy(Z, y, onehot=False):
    # 樣本的總數（有幾筆資料）
    m = len(Z)

    # 先把模型輸出 Z 丟進 softmax，算出每一類的機率
    F = softmax(Z)

    if onehot:
        # 如果 y 是 one-hot 編碼
        # 直接用 cross entropy 的公式算 loss
        loss = -np.sum(y * np.log(F)) / m
    else:
        # 如果 y 是類別的 index（例如 0、1、2）
        # 把 y 攤平成一維（但這行實際上不會影響後面）
        y.flatten()

        # 取出每筆資料「正確類別」對應的機率，再取 log
        log_Fy = -np.log(F[range(m), y])

        # 把所有樣本的 loss 加起來，再取平均
        loss = np.sum(log_Fy) / m

    return loss

def cross_entropy_grad_loss(Z, y, softmax_out=False, onehot=False):
    """
    優化後的函式：共用 Softmax 運算結果，同時回傳 Loss 與 Gradient。
    """
    # 1. 決定機率分布 F (Softmax 的結果)
    if softmax_out:
        # 如果已經是 Softmax 輸出，直接共用
        F = Z
    else:
        # 如果是 Logits，算一次 Softmax 就好，後面大家都用這份 F
        F = softmax(Z)

    # 2. 計算 Loss
    # 注意：這裡我們傳入已經算好的 F，並告訴後面的函式 softmax_out=True
    # (假設你原有的 softmax_cross_entropy 有支援這個判斷，或是我們直接在這裡算)
    m = len(y)
    if onehot:
        loss = -np.sum(y * np.log(F + 1e-12)) / m
    else:
        # 這裡示範直接計算，減少函式呼叫的開銷
        loss = -np.sum(np.log(F[np.arange(m), y] + 1e-12)) / m

    # 3. 計算梯度 (Gradient)
    # 直接利用剛剛算好的 F 進行計算，效能最優
    grad = F.copy()
    if onehot:
        grad = (grad - y) / m
    else:
        grad[np.arange(m), y] -= 1
        grad /= m

    return loss, grad

def rnn_loss_grad(Fs, Ys, loss_fn = cross_entropy_grad_loss, flatten = True):   
    # 總 loss 初始化
    loss = 0
    
    # 存每個時間點的輸出梯度 dF
    dFs = {}
   
    # 逐時間點計算
    for t in range(len(Fs)):
        F = Fs[t]      # 第 t 個時間點的預測輸出
        Y = Ys[t]      # 第 t 個時間點的正確答案
        
        # 如果需要就把答案攤平，避免 loss 計算出錯
        if flatten and Y.ndim >= 2:          
            Y = Y.flatten()
            
        # 計算該時間點的 loss 與對 F 的梯度
        loss_t, dF_t = loss_fn(F, Y)
        
        # 累加 loss
        loss += loss_t        
        
        # 存下該時間點的梯度
        dFs[t] = dF_t
       
    # 回傳總 loss 與各時間點的梯度
    return loss, dFs

# 梯度裁切：避免梯度爆炸
def grad_clipping(grads, alpha):
    # 算整包梯度的 L2 norm
    norm = math.sqrt(sum((grad ** 2).sum() for grad in grads))
    # 超過門檻就等比例縮小
    if norm > alpha:
        ratio = alpha / norm
        for i in range(len(grads)):
            grads[i] *= ratio 

def rnn_loss_grad(Fs, Ys, loss_fn = cross_entropy_grad_loss, flatten = True):   
    # 總 loss 初始化
    loss = 0
    
    # 存每個時間點的輸出梯度 dF
    dFs = {}
   
    # 逐時間點計算
    for t in range(len(Fs)):
        F = Fs[t]      # 第 t 個時間點的預測輸出
        Y = Ys[t]      # 第 t 個時間點的正確答案
        
        # 如果需要就把答案攤平，避免 loss 計算出錯
        if flatten and Y.ndim >= 2:          
            Y = Y.flatten()
            
        # 計算該時間點的 loss 與對 F 的梯度
        loss_t, dF_t = loss_fn(F, Y)
        
        # 累加 loss
        loss += loss_t        
        
        # 存下該時間點的梯度
        dFs[t] = dF_t
       
    # 回傳總 loss 與各時間點的梯度
    return loss, dFs
            
def rnn_backward(params, Xs, Hs, dZs, clip_value = 5.):    

    # 取出參數
    Wx, Wh, bh, Wf, bf = params
    
    # 初始化各參數梯度
    dWx, dWh, dWf = np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(Wf)
    dbh, dbf = np.zeros_like(bh), np.zeros_like(bf)       

    # 從下一個時間點傳回來的 hidden 梯度
    dh_next = np.zeros_like(Hs[0])
    h = Hs
    x = Xs
    
    T = len(Xs)  # time steps 數
    
    # 反向跑時間（BPTT）
    for t in reversed(range(T)): 
        dZ = dZs[t]  # 第 t 個時間點的輸出梯度
        
        # Output 層參數梯度
        dWf += np.dot(h[t].T, dZ)
        dbf += np.sum(dZ, axis=0, keepdims=True)         
        
        # 回傳到 hidden（包含當下 + 從未來傳回來的）
        dh = np.dot(dZ, Wf.T) + dh_next 
        
        # tanh 的梯度
        dZh = (1 - h[t] * h[t]) * dh 
        
        # Hidden 層參數梯度
        dbh += np.sum(dZh, axis=0, keepdims=True) 
        dWx += np.dot(x[t].T, dZh)
        dWh += np.dot(h[t-1].T, dZh)  # 用到前一個 hidden
        
        # 傳給前一個時間點
        dh_next = np.dot(dZh, Wh.T)
   
    # 打包梯度
    grads = [dWx, dWh, dbh, dWf, dbf]
    
    # 需要的話做 clipping
    if clip_value is not None:
        grad_clipping(grads, clip_value)
        
    return grads

class SGD():
    # ===== 最基本的 SGD + Momentum 實作 =====
    # 用來更新 model 參數的 optimizer

    def __init__(self, model_params, learning_rate=0.01, momentum=0.9):
        # model_params: 外部傳進來的參數列表（例如 [W1, b1, W2, b2, ...]）
        # learning_rate: 學習率，控制每次更新幅度
        # momentum: 動量係數，讓更新方向比較平滑，不會亂跳
        
        self.params = model_params
        self.lr = learning_rate
        self.momentum = momentum
        
        # vs: 用來存每個參數對應的「速度向量」
        # 跟 params 形狀一樣，初始化為 0
        self.vs = []
        for p in self.params:
            v = np.zeros_like(p)  # 建一個跟參數同 shape 的 0 矩陣
            self.vs.append(v)   
                

    def step(self, grads): 
        # ===== 做一次參數更新 =====
        # grads: 外部算好的梯度（通常是 backward 傳回來的）
        
        for i in range(len(self.params)):   
            grad = grads[i]  # 取出對應參數的梯度
            
            # Momentum 更新公式：
            # v = momentum * v + lr * grad
            # 前一輪的速度會保留一部分，加上這次的梯度
            self.vs[i] = self.momentum * self.vs[i] + self.lr * grad
            
            # 參數更新：
            # param = param - v
            # 用累積後的速度來調整參數
            self.params[i] -= self.vs[i]


    def scale_learning_rate(self, scale):
        # ===== 動態調整學習率 =====
        # 常見於 training 過程中做 lr decay
        # 例如：每幾個 epoch 把 lr 乘上 0.1
        self.lr *= scale

class AdaGrad():
    # ===== AdaGrad Optimizer 實作 =====
    # 核心概念：
    # 每個參數都有自己的學習率（自動調整）
    # 梯度大的參數，之後更新幅度會變小
    # 常見於 sparse feature（例如 NLP）場景

    def __init__(self, model_params, learning_rate=0.01):
        # model_params: 模型所有參數
        # learning_rate: 全域基礎學習率
        
        self.params = model_params
        self.lr = learning_rate
        
        # vs: 用來累積「梯度平方和」
        # 每個參數都會有一個對應的累積器
        self.vs = []
        
        # delta: 避免除以 0 的小常數（數值穩定用）
        self.delta = 1e-7
        
        for p in self.params:
            # 初始化為 0，shape 跟參數一樣
            v = np.zeros_like(p)
            self.vs.append(v)       
                 
    def step(self, grads): 
        # ===== 做一次參數更新 =====
        # grads: backward 計算好的梯度
        
        for i in range(len(self.params)):  
            grad = grads[i]
            
            # 累積梯度平方
            # v = v + grad^2
            # 這會讓常常出現大梯度的維度，v 變得很大
            self.vs[i] += grad**2
            
            # 參數更新公式：
            # param -= lr * grad / (sqrt(v) + delta)
            # v 越大，實際有效學習率越小
            # 等於是自動幫每個維度做 scaling
            self.params[i] -= self.lr * grad / (self.delta + np.sqrt(self.vs[i]))
            
    def scale_learning_rate(self, scale):
        # ===== 動態調整基礎學習率 =====
        # 有時還是會搭配 lr decay 使用
        self.lr *= scale

def mse_loss_grad(f, y):
    m = len(f)
    # 計算資料筆數（樣本數）

    loss = (1./m) * np.sum((f - y) ** 2)
    # 計算 MSE（均方誤差），用來衡量預測值跟實際值的差距

    grad = (2./m) * (f - y)
    # 計算 MSE 對預測值 f 的梯度，常用在梯度下降

    return loss, grad
    # 回傳 loss 值以及對應的梯度