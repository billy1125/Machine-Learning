import numpy as np
import math

def numerical_gradient(f, params, eps = 1e-6):
    numerical_grads = []
    for x in params:
        # x 可能是一個多維陣列（ndarray），針對其中的每個元素計算數值偏微分
        grad = np.zeros(x.shape)
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:            
            idx = it.multi_index
            old_value = x[idx]        
            x[idx] = old_value + eps  # 加上微小偏移量
            fx = f()  
            x[idx] = old_value - eps # 減去微小偏移量
            fx_ = f()
            grad[idx] = (fx - fx_) / (2*eps)  
            x[idx] = old_value      # 注意：一定要把這個權重參數還原回原本的值
            it.iternext()           # 繼續跑下一個元素的迭代            
      
        numerical_grads.append(grad)
    return numerical_grads

def numerical_gradient_from_df(f, p, df, h=1e-5):
  grad = np.zeros_like(p)
  it = np.nditer(p, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    idx = it.multi_index
    
    oldval = p[idx]
    p[idx] = oldval + h
    pos = f()       # 當依賴參數 p[idx] 變動後，重新呼叫 f() 計算輸出
    p[idx] = oldval - h
    neg = f()       # 當依賴參數 p[idx] 變動後，重新呼叫 f() 計算輸出
    p[idx] = oldval
       
    grad[idx] = np.sum((pos - neg) * df) / (2 * h)
    # grad[idx] = np.dot((pos - neg), df) / (2 * h)
    it.iternext()
  return grad
  
 
def mse_loss(F, Y, divid_2=False):
    m = F.shape[0]
    loss = np.sum((F - Y) ** 2) / m
    if divid_2: loss /= 2
    return loss
    
def mse_loss_grad(f, y):
    m = len(f)
    loss = (1./m) * np.sum((f-y)**2) # 計算均方誤差
    grad = (2./m) * (f-y)
    return loss, grad
  

def BCE_loss_grad(z, y, epsilon = 1e-8):
    m = len(y)
    assert(m > 0)
    f = sigmoid(z)    
    loss = -np.sum(y * np.log(f + epsilon) + (1 - y) * np.log(1 - f + epsilon))   
    grad = (f - y)
    loss /= m
    grad /= m
    return loss, grad
    

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def softmax(Z):
    # 減去最大值是為了防止 exp 數值爆炸（穩定性處理）
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
def softmax_backward(z, df):    
    f = softmax(z)
    grad = -np.outer(f, f) + np.diag(f.flatten())
    
    return df @ grad
    
def cross_entropy_loss(F, Y, onehot=False, epsilon = 1e-8):
    m = len(F) # 樣本數
    if onehot:
        return -(1./m) * np.sum(np.multiply(Y, np.log(F + epsilon)))
    else:         
        # F[i] 中對應 Y[i] 類別的 log 值
        return - (1./m) * np.sum(np.log(F[range(m), Y] + epsilon))

def softmax_cross_entropy(Z, y, onehot=False, epsilon = 1e-8):
    m = len(Z)
    F = softmax(Z)
    if onehot:
        loss = -np.sum(y * np.log(F)) / m
    else:
        y.flatten()
        log_Fy = -np.log(F[range(m), y] + epsilon)        
        loss = np.sum(log_Fy) / m
    return loss

def cross_entropy_grad(Z, Y, onehot = False, softmax_out=False):  
    if softmax_out:
        F = Z
    else:
        F = softmax(Z)
    if onehot:
        dZ = (F - Y) / len(Z)
    else:
        m = len(Y)
        dZ = F.copy()
        dZ[np.arange(m), Y] -= 1
        dZ /= m
        # I_i = np.zeros_like(Z)
        # I_i[np.arange(len(Z)), Y] = 1    
        # return (F - I_i) / len(Z)
    return dZ
    
def cross_entropy_grad_loss(F, y, softmax_out=False, onehot=False):
    if softmax_out:
        loss = cross_entropy_loss(F, y, onehot)
    else:    
        loss = softmax_cross_entropy(F, y, onehot)
    loss_grad = cross_entropy_grad(F, y, onehot, softmax_out)
    return loss, loss_grad

def data_iter(X, y, batch_size, shuffle=False):
    m = len(X)  
    indices = list(range(m))
    if shuffle:                 # 如果 shuffle 為 True，就隨機打亂索引順序
        np.random.shuffle(indices)
    for i in range(0, m - batch_size + 1, batch_size):
        batch_indices = np.array(indices[i: min(i + batch_size, m)])      
        yield X.take(batch_indices, axis=0), y.take(batch_indices, axis=0)   
        
        
def data_iterator(X, y, batch_size, shuffle=False):
    m = len(X)  
    indices = list(range(m))
    if shuffle:                 # 如果 shuffle 為 True，就打亂順序
        np.random.shuffle(indices)
    for i in range(0, m - batch_size + 1, batch_size):
        batch_indices = np.array(indices[i: min(i + batch_size, m)])      
        yield X.take(batch_indices, axis=0), y.take(batch_indices, axis=0)   
    
    
def rnn_loss_grad(Fs, Ys, loss_fn = cross_entropy_grad_loss, flatten = True):
    loss = 0
    dFs = {}
   
    for t in range(len(Fs)):
        F = Fs[t]
        Y = Ys[t]   
        if flatten and Y.ndim >= 2:
            Y = Y.flatten() # 將標籤攤平為一維
        loss_t, dF_t = loss_fn(F, Y)
        loss += loss_t        
        dFs[t] = dF_t
    return loss, dFs

def one_hot_idx(idx, vocab_size):
    x = np.zeros((1, vocab_size))
    x[0, idx] = 1
    return x
    
def clip_grad_norm_nn(parameters, max_norm, norm_type=2): 
    # 梯度剪裁（Gradient Clipping），防止梯度爆炸
    max_norm = float(max_norm)    
    if norm_type is None:
        total_norm = max(max(grad.max(), abs(grad.min())) for p, grad in parameters)
    else: 
        norm_type = float(norm_type)
        total_norm = 0.  
        total_norm = math.sqrt(sum((grad ** norm_type).sum() for p, grad in parameters))
        total_norm = total_norm ** (1. / norm_type)
        
    ratio = max_norm / (total_norm + 1e-6)    
    if ratio < 1:        
        for p, grad in parameters:
            grad[:] = grad * ratio
            


def data_iterator_X(X, batch_size, shuffle = True, repeat=False):
    m = len(X)   
    indices = list(range(m))
    while True:
        if shuffle:
            np.random.shuffle(indices) 
        for i in range(0, m, batch_size):
            if i + batch_size > m:
                break
            j = np.array(indices[i: i + batch_size])
            yield X.take(j, axis=0)  
        if not repeat:
            return