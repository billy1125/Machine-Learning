import numpy as np
import math
import datetime
DEBUG = False

#----------------- 優化器 (Optimizer) -------------------------
class Optimizer():
    def __init__(self):
        self.params = None
        
    def reset(self):
        pass

    def zero_grad(self):  
        """將所有參數的梯度歸零，避免梯度累積"""
        if self.params is None: return
        for i, _ in enumerate(self.params):
             # 將參數列表中的梯度部分（索引 1）清空為 0
            self.params[i][1][:] = 0.  
     
    def step(self): 
        pass
    
    def regularization(self, reg):
        """計算正則化（L2 Regularization），並將其加入梯度中"""
        reg_loss = 0
        for p, grad in self.params:
             # 更新梯度：加入正則化項的導數
             grad[:] = grad + 2 * reg * p
             reg_loss += np.sum(p**2)
        return reg * reg_loss

    def loss_reg(self, reg):
        """單純計算正則化的 Loss 值"""
        reg_loss = 0
        for p, grad in self.params:            
             reg_loss += np.sum(p**2)
        return reg * reg_loss
        
    def parameters(self):
        return self.params  
    
class SGD(Optimizer):
    """隨機梯度下降法 (Stochastic Gradient Descent)"""
    def __init__(self, model_params, learning_rate=0.01, momentum=0.9, decay_every=None, decay=0.9):
        super().__init__()
        self.params, self.lr, self.momentum = model_params, learning_rate, momentum
        self.vs = [] # 儲存動量 (velocity) 的清單
        for p, grad in self.params:
            v = np.zeros_like(p)
            self.vs.append(v)
            
        self.iter = 1
        self.decay_every, self.decay = decay_every, decay

    def reset(self):
        self.iter = 1
        for p in self.vs:
            p[:] = 0.

    def step(self): 
        """執行參數更新"""
        for i, _ in enumerate(self.params):     
            p, grad = self.params[i]            
            # 計算帶動量的更新值
            self.vs[i] = self.momentum * self.vs[i] + self.lr * grad             
            self.params[i][0] -= self.vs[i]
            
        self.iter += 1
        # 如果有設定 decay_every，就定期調降學習率 (Learning Rate Decay)
        if self.decay_every is not None and self.iter % self.decay_every == 0:
            self.lr *= self.decay

    def scale_learning_rate(self, scale):
        self.lr *= scale
        
    def debug_params(self):
         if DEBUG:
            for p, grad in self.params:  
                print("參數 p:", p)
                print("梯度 grad:", grad)
            print()

class Adam(Optimizer):
    """Adam 優化器：結合 AdaGrad 與 Momentum 的優點"""
    def __init__(self, model_params, learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        super().__init__()
        
        self.params, self.lr = model_params, learning_rate
        self.beta_1, self.beta_2, self.epsilon = beta_1, beta_2, epsilon
        self.ms = [] # 一階動量 (梯度均值)
        self.vs = [] # 二階動量 (梯度平方均值)
        self.t = 0
        for p, grad in self.params:
            m = np.zeros_like(p)
            v = np.zeros_like(p)
            self.ms.append(m)
            self.vs.append(v)

    def reset(self):
        for i in range(len(self.vs)):
            self.ms[i] = 0.
            self.vs[i] = 0.

    def step(self):   
        beta_1, beta_2, lr = self.beta_1, self.beta_2, self.lr
        self.t += 1
        t = self.t
        for i, _ in enumerate(self.params):     
            p, grad = self.params[i]       
            
            # 更新一階與二階動量
            self.ms[i] = beta_1 * self.ms[i] + (1 - beta_1) * grad
            self.vs[i] = beta_2 * self.vs[i] + (1 - beta_2) * grad**2        
            
            # 偏差修正 (Bias Correction)
            m_1 = self.ms[i] / (1 - np.power(beta_1, t))
            v_1 = self.vs[i] / (1 - np.power(beta_2, t))  
            
            # 更新參數
            self.params[i][0] -= lr * m_1 / (np.sqrt(v_1) + self.epsilon)
      
    def scale_learning_rate(self, scale):
        self.lr *= scale

class AdaGrad(Optimizer):
    """AdaGrad：自適應調整每個參數的學習率"""
    def __init__(self, model_params, learning_rate=0.01, momentum=0.9):
        super().__init__()
        self.params, self.lr, self.momentum = model_params, learning_rate, momentum
        self.vs = []
        self.delta = 1e-7
        for p, grad in self.params:
            v = np.zeros_like(p)
            self.vs.append(v)
        
    def step(self): 
        for i, _ in enumerate(self.params):     
            p, grad = self.params[i]
            # 累積梯度的平方
            self.vs[i] += grad**2
            # 根據歷史梯度的總和來調整更新步長
            self.params[i][0] -= self.lr * grad / (self.delta + np.sqrt(self.vs[i]))

    def scale_learning_rate(self, scale):
        self.lr *= scale

class RMSprop(Optimizer):
    """RMSprop：解決 AdaGrad 學習率遞減過快的問題"""
    def __init__(self, model_params, learning_rate=0.01, beta=0.9, epsilon=1e-8):
        super().__init__()
        self.params, self.lr, self.beta, self.epsilon = model_params, learning_rate, beta, epsilon
        self.vs = []        
        for p, grad in self.params:
            v = np.zeros_like(p)
            self.vs.append(v)

    def step(self): 
        beta = self.beta
        for i, _ in enumerate(self.params):     
            p, grad = self.params[i] 
            v = self.vs[i]
            # 使用指數加權移動平均更新平方梯度
            v[:] = beta * v + (1 - beta) * grad**2         
            p[:] = p - (self.lr * grad / (np.sqrt(v) + self.epsilon))            
          
    def scale_learning_rate(self, scale):
        self.lr *= scale

def data_iterator(X, y, batch_size, shuffle=False):
    """資料迭代器，負責將資料切分成一個個 Batch"""
    m = len(X)  
    indices = list(range(m))
    if shuffle: # 打亂資料順序
        np.random.shuffle(indices)
    for i in range(0, m - batch_size + 1, batch_size):
        batch_indices = np.array(indices[i: min(i + batch_size, m)])      
        yield X.take(batch_indices, axis=0), y.take(batch_indices, axis=0)

def train_nn(nn, X, y, optimizer, loss_fn, epochs=100, batch_size=50, reg=1e-3, print_n=10, save=False):
    """神經網路訓練主迴圈"""
    iter = 0
    losses = [] 
    today = datetime.datetime.now().strftime("%Y-%m-%d-%p-%I-%M") 
    
    for epoch in range(epochs):
        for X_batch, y_bacth in data_iterator(X, y, batch_size):     
            # 1. 梯度清零
            optimizer.zero_grad()      
          
            # 2. 前向傳播 (Forward Pass)
            f = nn(X_batch) 
            loss, loss_grad = loss_fn(f, y_bacth)       
            
            # 3. 反向傳播 (Backward Pass)
            nn.backward(loss_grad, reg)               
            loss += nn.reg_loss(reg)
           
            # 4. 更新參數
            optimizer.step()

            losses.append(loss)            
            if iter % print_n == 0:
                print('[反覆次數 %5d, 第 %d 個 Epoch] loss: %.3f' % (iter + 1, epoch + 1, loss))                
                if save:
                    # 儲存目前的權重參數
                    nn.save_parameters(today + '-train_nn.npy')
            iter += 1          

    return losses

def grad_clipping(grads, alpha):
    """梯度剪裁：防止梯度爆炸 (Gradient Explosion)"""
    norm = math.sqrt(sum((grad ** 2).sum() for grad in grads))
    if norm > alpha:
        ratio = alpha / norm
        for i in range(len(grads)):
            grads[i] *= ratio