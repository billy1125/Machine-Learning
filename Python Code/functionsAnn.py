import numpy as np

def sigmoid(x):
    # Sigmoid 函數，常用在二元分類，將輸入值壓到 0~1 之間
    return 1 / (1 + np.exp(-x))

def binary_cross_entropy(f, y, epsilon=1e-8):
    # 二元交叉熵損失函數
    # f: 模型預測值 (機率)
    # y: 真實標籤 (0 或 1)
    # epsilon: 避免 log(0) 用的小常數

    # 原本可能是想逐筆算 loss（被註解掉）
    # np.sum(y*np.log(f+epsilon)+ (1-y)*np.log(1-f+epsilon), axis=1)

    m = len(y)  # 資料筆數
    # 計算整體平均 loss
    return - (1./m) * np.sum(
        np.multiply(y, np.log(f + epsilon)) +
        np.multiply((1 - y), np.log(1 - f + epsilon))
    )

def binary_cross_entropy_grad(out, y, sigmoid_out=True, epsilon=1e-8):
    # 計算 binary cross entropy 對輸出值的梯度
    # out: 模型輸出
    # y: 真實標籤
    # sigmoid_out: 表示 out 是否已經過 sigmoid

    if sigmoid_out:
        f = out  # 如果已經是 sigmoid 輸出，直接使用
        grad = ((f - y) / (f * (1 - f) + epsilon)) / (len(y))
        # f = np.clip(f, eps, 1 - eps) # 建議寫法
        # grad = ((f - y) / (f * (1 - f))) / (len(y))
    else:
        f = sigmoid(out)  # 如果 out 是 z，先做 sigmoid
        grad = (f - y) / (len(y))

def binary_cross_entropy_loss_grad(out, y, sigmoid_out=True, epsilon=1e-8):
    # 同時計算 loss 與 gradient
    # 常用在訓練時一次回傳兩個結果

    if sigmoid_out:
        f = out  # 已經是 sigmoid 結果
        grad = ((f - y) / (f * (1 - f) + epsilon)) / (len(y))
        # f = np.clip(f, eps, 1 - eps) # 建議寫法
        # grad = ((f - y) / (f * (1 - f))) / (len(y))
    else:
        f = sigmoid(out)  # out 是尚未經過 sigmoid 的值
        grad = (f - y) / (len(y))

    # 計算 binary cross entropy loss
    loss = binary_cross_entropy(f, y, epsilon)

    return loss, grad

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

def cross_entropy_grad(Z, Y, onehot=False, softmax_out=False):
    # 判斷輸入的 Z 是不是已經是 softmax 的輸出
    if softmax_out:
        F = Z
    else:
        # 如果不是，就先做一次 softmax
        F = softmax(Z)

    if onehot:
        # 如果 Y 是 one-hot 編碼，直接用 F - Y
        grad = (F - Y) / len(Z)
    else:
        # 如果 Y 是類別索引（例如 [0, 2, 1]）
        m = len(Y)          # 樣本數
        grad = F.copy()       # 複製一份避免動到原本的 F
        # 對正確類別的位置減 1
        grad[np.arange(m), Y] -= 1
        # 對 batch size 做平均
        grad /= m

        # 以下是另一種寫法，這裡先註解起來
        # I_i = np.zeros_like(Z)
        # I_i[np.arange(len(Z)),Y] = 1
        # return (F - I_i) /len(Z)  #Z.shape[0]

    # 回傳對 Z 的梯度
    return grad

def cross_entropy_grad_loss(F, y, softmax_out=False, onehot=False):
    # 判斷輸入的 F 是否已經經過 Softmax 處理
    if softmax_out:
        loss = softmax_cross_entropy(F, y, onehot)
        # 註解掉的這行可能是想區分一般的 Cross Entropy 與 Softmax 版
        # loss = cross_entropy_loss(F, y, onehot)
    else:
        # 如果 F 還是 Logits（原始輸出），則執行含 Softmax 的 Cross Entropy 計算
        loss = softmax_cross_entropy(F, y, onehot)

    # 計算梯度（Grad），同樣會根據 softmax_out 決定內部是否要補做 Softmax
    grad = cross_entropy_grad(F, y, onehot, softmax_out)

    # 同時回傳損失值與梯度，方便後向傳播使用
    return loss, grad

# def cross_entropy_grad_loss(Z, y, softmax_out=False, onehot=False):
#     """
#     優化後的函式：共用 Softmax 運算結果，同時回傳 Loss 與 Gradient。
#     """
#     # 1. 決定機率分布 F (Softmax 的結果)
#     if softmax_out:
#         # 如果已經是 Softmax 輸出，直接共用
#         F = Z
#     else:
#         # 如果是 Logits，算一次 Softmax 就好，後面大家都用這份 F
#         F = softmax(Z)

#     # 2. 計算 Loss
#     # 注意：這裡我們傳入已經算好的 F，並告訴後面的函式 softmax_out=True
#     # (假設你原有的 softmax_cross_entropy 有支援這個判斷，或是我們直接在這裡算)
#     m = len(y)
#     if onehot:
#         loss = -np.sum(y * np.log(F + 1e-12)) / m
#     else:
#         # 這裡示範直接計算，減少函式呼叫的開銷
#         loss = -np.sum(np.log(F[np.arange(m), y] + 1e-12)) / m

#     # 3. 計算梯度 (Gradient)
#     # 直接利用剛剛算好的 F 進行計算，效能最優
#     grad = F.copy()
#     if onehot:
#         grad = (grad - y) / m
#     else:
#         grad[np.arange(m), y] -= 1
#         grad /= m

#     return loss, grad

def mse_loss_grad(f, y):
    m = len(f)
    # 計算資料筆數（樣本數）

    loss = (1./m) * np.sum((f - y) ** 2)
    # 計算 MSE（均方誤差），用來衡量預測值跟實際值的差距

    grad = (2./m) * (f - y)
    # 計算 MSE 對預測值 f 的梯度，常用在梯度下降

    return loss, grad
    # 回傳 loss 值以及對應的梯度

def dRelu(x):
    # ReLU 的微分：x > 0 回傳 1，否則回傳 0
    return 1 * (x > 0)

def max_abs(s):
    # 計算一組矩陣或向量中，所有元素的最大絕對值
    max_value = 0
    for x in s:
        max_value_ = np.max(np.abs(x))  # 取單一矩陣的最大絕對值
        if(max_value_ > max_value):
            max_value = max_value_
    return max_value

# 產生一個非線性可分（Non-linearly separable）的「旋渦狀」資料集
def gen_spiral_dataset(N=100, D=2, K=3):
    # 初始化特徵矩陣 X，形狀為 (樣本總數, 維度)，這裡預設產生 300 筆 2 維資料
    X = np.zeros((N*K, D))  # 建立資料矩陣，每一列代表一筆資料

    # 初始化標籤陣列 y，記錄每筆資料屬於哪一個類別（0, 1, 2）
    y = np.zeros(N*K, dtype='uint8')  # 建立標籤陣列，用來存每筆資料的類別

    for j in range(K):  # 依序產生每一個類別 (Class) 的資料
        # 計算目前類別在矩陣中對應的索引範圍 (Index range)
        ix = range(N*j, N*(j+1))

        # 產生半徑 r，從 0 到 1 均勻分布，代表點從原點往外擴散
        r = np.linspace(0.0, 1, N)  # 半徑，從中心往外均勻增加

        # 產生角度 t，隨類別不同有不同的起始角度，並加入高斯雜訊 (Noise) 讓資料不會太死板
        t = np.linspace(j*4, (j+1)*4, N) + np.random.randn(N)*0.2  # 角度，加上一點雜訊讓資料比較分散

        # 利用極座標轉直角座標 (r*sin(t), r*cos(t))，並用 np.c_ 進行欄位合併
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]  # 轉成平面座標，形成螺旋狀

        # 填入對應的類別編號
        y[ix] = j  # 設定對應的類別標籤

    return X, y  # 回傳產生的特徵矩陣 X 與目標標籤 y

def numerical_gradient_from_df(f, p, df, h=1e-5):
  # 建立一個與 p 形狀相同、內容全為 0 的陣列，用來存每個參數的梯度
  grad = np.zeros_like(p)

  # 使用 nditer 逐一走訪 p 中的每一個元素（支援多維陣列）
  it = np.nditer(p, flags=['multi_index'], op_flags=['readwrite'])

  # 只要 iterator 還沒跑完就持續計算
  while not it.finished:
    # 取得目前走訪到的索引位置
    idx = it.multi_index

    # 先把原本的參數值存起來
    oldval = p[idx]

    # 將該參數往正方向微調一點
    p[idx] = oldval + h
    pos = f()       # 在參數被改動後重新呼叫 f()，取得正方向的輸出結果

    # 將該參數往負方向微調一點
    p[idx] = oldval - h
    neg = f()       # 在參數被改動後重新呼叫 f()，取得負方向的輸出結果

    # 將參數值還原成原本的狀態，避免影響下一次計算
    p[idx] = oldval

    # 使用中央差分法計算梯度，並與上游傳下來的 df 做加權
    grad[idx] = np.sum((pos - neg) * df) / (2 * h)

    # 另一種寫法（使用內積），目前被註解掉
    # grad[idx] = np.dot((pos - neg), df) / (2 * h)

    # 移動到下一個參數位置
    it.iternext()

  # 回傳整個參數 p 的數值梯度
  return grad

# 通用數值梯度函式
def numerical_gradient(f, params, eps=1e-6):
    numerical_grads = []  # 用來存每一個參數對應的梯度

    for x in params:
        # x 可能是多維陣列，這裡會針對 x 裡面的每一個元素計算數值梯度
        grad = np.zeros(x.shape)  # 建立一個跟 x 形狀一樣的陣列來存梯度

        # 使用 nditer 逐一走訪 x 裡面的每個元素
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

        while not it.finished:
            idx = it.multi_index   # 目前元素的索引位置

            old_value = x[idx]     # 先把原本的值存起來

            x[idx] = old_value + eps  # 對目前的值加上一個很小的 eps
            fx = f()                  # 計算 f(x + eps) 的結果

            x[idx] = old_value - eps  # 對目前的值減去一個很小的 eps
            fx_ = f()                 # 計算 f(x - eps) 的結果

            # 使用中央差分法計算數值梯度
            grad[idx] = (fx - fx_) / (2 * eps)

            x[idx] = old_value        # 記得把參數值還原，避免影響後續計算
            it.iternext()             # 移動到下一個元素

        numerical_grads.append(grad)  # 將目前參數的梯度存起來

    return numerical_grads            # 回傳所有參數的梯度

# def f():
#     return compute_loss_reg(forward_propagation, softmax_cross_entropy_reg, X, y, parameters)

def data_iter(X, y, batch_size, shuffle=False):
    m = len(X)  
    indices = list(range(m))
    
    # 如果 shuffle 為 True，則隨機打亂索引順序，增加模型訓練的泛化能力
    if shuffle:
        np.random.shuffle(indices)
        
    # 以 batch_size 為步長進行迭代，生成每一個批次 (Batch) 的資料
    for i in range(0, m - batch_size + 1, batch_size):
        # 根據目前的位移量選取對應的索引，並確保不會超出總資料筆數
        batch_indices = np.array(indices[i: min(i + batch_size, m)])
        
        # 使用 yield 產生 Generator（產生器），節省記憶體空間
        # 透過 take 根據索引從原始資料中取出該批次的特徵 X 與標籤 y
        yield X.take(batch_indices, axis=0), y.take(batch_indices, axis=0)

   
def train_batch(nn, XX, YY, loss_function, epochs=10000, batch_size=50, learning_rate=1e-0, reg=1e-3, print_n=10, softmax_out=False, onehot=False):
    iter = 0  # 紀錄總迭代次數
    losses = []  # 用來存每次印出時的損失值
    for epoch in range(epochs):
        # 每一輪訓練 (Epoch) 都重新打亂資料，取出每一批次的 X 與 y
        for X, y in data_iter(XX, YY, batch_size, True):             
            # 1. 前向傳播 (Forward Pass)：計算神經網路的預測輸出
            f = nn.forward(X)        
            
            # 2. 計算損失值 (Loss) 與對輸出值的梯度 (loss_grad)
            loss, loss_grad = loss_function(f, y, softmax_out=softmax_out, onehot=onehot)        
            
            # 3. 加入正規化項 (Regularization Loss) 以防止過擬合 (Overfitting)
            loss += nn.reg_loss(reg)

            # 4. 反向傳播 (Backward Pass)：傳入損失梯度並計算權重梯度
            nn.backward(loss_grad, reg)

            # 5. 更新參數 (Parameter Update)：根據學習率調整權重與偏置項
            nn.update_parameters(learning_rate)

            # 定期印出訓練資訊，方便監控模型收斂狀況
            if iter % print_n == 0:
                print("iteration %d: loss %f" % (iter, loss)) 
            iter += 1

            losses.append(loss)  # 紀錄每次的損失值
            
    return losses  # 回傳所有紀錄的損失值