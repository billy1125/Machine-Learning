import numpy as np
import math
import matplotlib.pyplot as plt
import functionsRnn as fr


def rnn_train_epoch(params, data_iter, optimizer, iterations, loss_function, print_n=100):
    # ===== 跑一個 epoch（或跑到指定 iterations 就停） 的 RNN 訓練迴圈 =====
    # params: [Wx, Wh, bh, Wf, bf] 參數包
    # data_iter: data loader / iterator，每次吐一個 batch 的 (Xs, Ys, start)
    # optimizer: 例如 SGD / AdaGrad，負責更新 params
    # iterations: 最多跑幾個 step（用來限制訓練長度）
    # loss_function: 給定輸出與標籤，回傳 (loss, dZs)
    # print_n: 每隔幾步印一次 loss

    # 把參數拆開來，方便後面取 shape 或 debug
    Wx, Wh, bh, Wf, bf = params

    # 用來記錄每一步的 loss（之後可畫圖或看收斂狀況）
    losses = []

    # iter: 手動的 step counter（不是 Python 內建 iter，那個名字被你覆蓋掉了）
    iter = 0

    # hidden_size: hidden state 維度，通常從 Wh 的 shape 推
    hidden_size = Wh.shape[0]

    # ===== 逐 batch 訓練 =====
    for Xs, Ys, start in data_iter:

        # batch_size 通常看 Xs[0] 的第一維（時間步 0 的 batch）
        batch_size = Xs[0].shape[0]

        # start 代表「這是新序列的開頭」或「要重置 hidden state」
        # 常見於 truncated BPTT 或分段讀資料時，用來決定要不要清空狀態
        if start:
            H = fr.rnn_hidden_state_init(batch_size, hidden_size)

        # ===== Forward Pass =====
        # Zs: 每個時間步的輸出（可能是 logits / scores）
        # Hs: 每個時間步的 hidden state（用來做 backward）
        Zs, Hs = fr.rnn_forward(params, Xs, H)

        # ===== 計算 Loss + 對輸出梯度 =====
        # dzs: loss 對 Zs 的梯度（後面丟進 backward 當起點）
        loss, dzs = loss_function(Zs, Ys)

        # ===== Debug 區（目前被關掉）=====
        # 需要看 shape 或中間值時，把 if False 改成 if True 就好
        if False:
            print("Z.shape", Zs[0].shape)
            print("Y.shape", Ys[0].shape)
            print("H", H.shape)

        # ===== Backward Pass =====
        # 用 BPTT 算出各參數的梯度
        dWx, dWh, dbh, dWf, dbf = fr.rnn_backward(params, Xs, Hs, dzs)

        # ===== 更新 hidden state，讓下一個 batch 可以接著用 =====
        # 這邊取倒數第二個（len(Hs)-2）當作「最後時刻」hidden state
        H = Hs[len(Hs)-2]

        # 把梯度包起來丟給 optimizer
        grads = [dWx, dWh, dbh, dWf, dbf]
        optimizer.step(grads)

        # 記錄 loss
        losses.append(loss)

        # 依照 print_n 的頻率印訓練狀況，方便看有沒有收斂或爆炸
        if iter % print_n == 0:
            print('iter %d, loss: %f' % (iter, loss))

        iter += 1

        # 超過指定 iterations 就提早收工（等於只跑部分 epoch）
        if iter > iterations:
            break

    # 回傳：整段訓練的 loss list + 最後的 hidden state（方便接續訓練或做 stateful RNN）
    return losses, H

# ===== 程式概要說明 =====
# 本函式是 RNN 用的「連續取樣資料產生器（consecutive sampling）」加強版：
import numpy as np

def rnn_data_iter_consecutive(data, batch_size, seq_len, start_range=10, to_3D=True):
    # 每次從不同的起點開始取樣，讓每個 epoch 的樣本略有不同
    start = np.random.randint(0, start_range)
    
    # 計算每個 batch 區塊可用的長度（-1 是因為 Y 要右移一格）
    block_len = (len(data) - start - 1) // batch_size

    # 取出可整除 batch_size 的連續資料作為 X / Y
    Xs = data[start:start + block_len * batch_size]
    Ys = data[start + 1:start + block_len * batch_size + 1]

    # reshape 成 (batch_size, block_len)：每列是一條連續序列
    Xs = Xs.reshape(batch_size, -1)
    Ys = Ys.reshape(batch_size, -1)

    # 計算每條序列可切出幾段 seq_len
    reset = True
    num_batches = Xs.shape[1] // seq_len

    # 逐段切出長度為 seq_len 的樣本
    for i in range(0, num_batches * seq_len, seq_len):
        X = Xs[:, i:(i + seq_len)]
        Y = Ys[:, i:(i + seq_len)]

        # 依需求轉成 RNN 常用格式
        if to_3D:
            # 先把維度從 (batch_size, seq_len) 換成 (seq_len, batch_size)
            X = np.swapaxes(X, 0, 1)
            Y = np.swapaxes(Y, 0, 1)

            # 再補上一個 feature 維度 => (seq_len, batch_size, feature_dim)
            X = X.reshape(X.shape[0], X.shape[1], -1)
            Y = Y.reshape(Y.shape[0], Y.shape[1], -1)
        else:
            # 只做 (batch_size, seq_len) -> (seq_len, batch_size)
            X = np.swapaxes(X, 0, 1)
            Y = np.swapaxes(Y, 0, 1)

        # 第一段資料回傳 reset=True，後續為 False
        if reset:
            reset = False
            yield X, Y, True
        else:
            yield X, Y, False



# 產生時間序列資料（sin/cos 混合波）
T = 5000  # 總點數
time = np.arange(0, T)
data = np.sin(time * 0.1) + np.cos(time * 0.2)
print(data.shape)

plt.figure(figsize=(10, 4))
# plt.plot(time, data)
plt.plot(time[:200], data[:200])
plt.title("Time Series Data")
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)

plt.show()

# ===== 程式概要說明 =====
# 這段程式用「人造的週期訊號（sin + cos）」當作時間序列資料，
# 來示範如何用自製的 RNN（搭配前面寫的 rnn_data_iter_consecutive）做訓練。
#
# 主要流程：
# 1️⃣ 先產生長度 T 的時間序列 data（sin/cos 混合，當作可預測的波形）
# 2️⃣ 設定 RNN 超參數：batch_size、hidden_size、seq_length 等
# 3️⃣ 初始化 RNN 參數 params 與 hidden state H
# 4️⃣ 用 rnn_data_iter_consecutive 把資料切成 (X, Y, reset)：
#    - X：長度 seq_length 的輸入序列
#    - Y：對應右移 1 步的目標序列
# 5️⃣ 先跑一次 forward 與 loss，檢查維度是否正確
# 6️⃣ 進入訓練迴圈：每個 epoch 重新產生 iterator，跑固定 iterations 更新參數
# 7️⃣ 每個 epoch 把 loss 取平均後存起來（losses 用來看訓練趨勢）

# ===== RNN 參數設定 =====
batch_size = 3
input_dim = 1
output_dim = 1
hidden_size = 100
seq_length = 50

# 初始化 RNN 參數與 hidden state
params = fr.rnn_params_init(input_dim, hidden_size, output_dim)
H = fr.rnn_hidden_state_init(batch_size, hidden_size)

# 建立資料產生器（start_range=2：每次 epoch 起點只會是 0 或 1）
data_it = rnn_data_iter_consecutive(data, batch_size, seq_length, 2)

# 取一個 batch 出來確認 shape
x, y, _ = next(data_it)
print("X:", x.shape, "Y:", y.shape, "H:", H.shape)

# 定義 loss function（這裡用 MSE 的 gradient）
loss_function = lambda F, Y: fr.rnn_loss_grad(F, Y, fr.mse_loss_grad, False)

# ===== 先跑一次 forward 檢查流程 =====
Zs, Hs = fr.rnn_forward(params, x, H)
print("Z:", Zs[0].shape, "H:", Hs[0].shape)

# 計算 loss 與對輸出 Z 的梯度 dzs
loss, dzs = loss_function(Zs, y)
print(dzs[0].shape)

# ===== 訓練超參數 =====
epoches = 10
learning_rate = 5e-4
iterations = 200
losses = []

# 選擇 optimizer（SGD + momentum）
momentum = 0.9
optimizer = fr.SGD(params, learning_rate, momentum)

# ===== 開始訓練 =====
for epoch in range(epoches):
    # 每個 epoch 重建 iterator（start_range=100：起點更隨機）
    data_it = rnn_data_iter_consecutive(data, batch_size, seq_length, 100)

    # 跑一個 epoch（更新 iterations 次）
    epoch_losses, H = rnn_train_epoch(
        params,
        data_it,
        optimizer,
        iterations,
        loss_function,
        print_n=50
    )

    # 取這個 epoch 的 loss 平均值當作代表
    epoch_losses = np.array(epoch_losses).mean()
    losses.append(epoch_losses)