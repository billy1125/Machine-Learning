import numpy as np

def row2im(dx_row, oH, oW, kH, kW, S):
    """
    將矩陣化的 Row 向量還原回圖片格式（用於反向傳播時，將梯度還原回原始影像維度）
    """
    nRow, K2C = dx_row.shape[0], dx_row.shape[1]
    C = K2C // (kH * kW)  # 計算通道數 (Channels)
    N = nRow // (oH * oW)  # 計算樣本數 (Batch Size)
    oSize = oH * oW        # 輸出特徵圖的總像素點
    H = (oH - 1) * S + kH  # 計算還原後的原始高度
    W = (oW - 1) * S + kW  # 計算還原後的原始寬度
    dx = np.zeros([N, C, H, W])
    
    for i in range(oSize):
        # 根據步長 (Stride) 抓出 N 個樣本對應的行向量
        row = dx_row[i::oSize, :]         
        h_start = (i // oW) * S
        w_start = (i % oW) * S     
        # 將梯度累加回對應的圖片區域（處理重疊區域的梯度加總）
        dx[:, :, h_start:h_start+kH, w_start:w_start+kW] += row.reshape((N, C, kH, kW))
    return dx

# row2im 的另一個版本，邏輯相似但透過雙層迴圈處理空間維度
def row2im_v2(dx_row, oH, oW, kH, kW, S):
    nRow, K2C = dx_row.shape[0], dx_row.shape[1]
    C = K2C // (kH * kW)
    N = nRow // (oH * oW)    
    oSize = oH * oW  
    
    H = (oH - 1) * S + kH
    W = (oW - 1) * S + kW
    dx = np.zeros([N, C, H, W])
    for h in range(oH):
        hS = h * S
        hS_kH = hS + kH
        h_start = h * oW
        for w in range(oW):
            wS = w * S          
            # 根據目前的索引位置取出對應的 row
            row = dx_row[h_start+w::oSize, :]                    
            dx[:, :, hS:hS_kH, wS:wS+kW] += row.reshape(N, C, kH, kW)          
    return dx

# im2row 的第一個版本，使用雙層迴圈遍歷空間維度
# 這個版本的效率較低，因為 Python 的迴圈性能不佳，實際使用中建議改用基於索引的版本 (im2row_indices)
# 參數：x: 輸入圖片 (N, C, H, W), kH: 卷積核高度, kW: 卷積核寬度, S: 步長 (Stride)
def im2row(x, kH, kW, S=1):
    # N: 樣本數, C: 通道數, H: 影像高度, W: 影像寬度
    N, C, H, W = x.shape
    
    # 計算輸出特徵圖 (Feature Map) 的空間維度
    oH = (H - kH) // S + 1
    oW = (W - kW) // S + 1
    
    # 初始化一個空的矩陣來存放結果
    # 列數 (Rows)：總共會取出的區塊數量 (N * 輸出高度 * 輸出寬度)
    # 欄數 (Cols)：每個區塊攤平後的總像素點 (卷積核高 * 卷積核寬 * 通道數)
    row = np.empty((N * oH * oW, kH * kW * C))
    
    oSize = oH * oW  # 單張圖片產生的區塊總數
    
    # 透過雙層迴圈遍歷所有輸出的座標點 (h, w)
    for h in range(oH):
        hS = h * S           # 窗口在原始影像上的垂直起始位置
        hS_kH = hS + kH      # 窗口在原始影像上的垂直結束位置
        h_start = h * oW     # 這是為了定位目前在 oSize 中的相對索引
        
        for w in range(oW):
            wS = w * S       # 窗口在原始影像上的水平起始位置
            
            # 1. 擷取區塊 (Patch)：從所有樣本中取出該位置的窗口
            # patch 的維度維 (N, C, kH, kW)
            patch = x[:, :, hS:hS_kH, wS:wS + kW]
            
            # 2. 攤平並填入矩陣：
            # 利用步進切片 (Step Slicing) [h_start+w::oSize] 
            # 將 N 個樣本的相同位置區塊，分散填入對應的 Row
            # 例如：row[0] 是第一個樣本的區塊1, row[oSize] 是第二個樣本的區塊1
            row[h_start + w::oSize, :] = np.reshape(patch, (N, -1))
            
    return row

def conv_forward(X, K, S=1, P=0):
    """
    卷積層的前向傳播 (Forward Pass)
    """
    N, C, H, W  = X.shape      # 輸入資料的維度 (Batch, Channel, Height, Width)
    F, C_k, kH, kW = K.shape   # 卷積核 (Kernel/Filter) 維度
    
    # 處理補零 (Padding)
    if P == 0:
        X_pad = X
    else:
        # 在 H 與 W 維度進行常數補零
        X_pad = np.pad(X, ((0, 0), (0, 0), (P, P), (P, P)), 'constant')
   
    # 核心優化：將圖片區塊轉換成矩陣行 (im2row)，方便後續進行矩陣運算 (Matrix Multiplication)
    X_row = im2row(X_pad, kH, kW, S)
    
    # 將卷積核攤平 (Flatten) 並轉置，準備進行點積
    K_col = K.reshape(K.shape[0], -1).transpose()    
    
    # 執行矩陣相乘，這是卷積運算效能最高的核心步驟
    Z_row = np.dot(X_row, K_col)
    
    # 計算輸出特徵圖 (Feature Map) 的空間維度
    oH = (X_pad.shape[2] - kH) // S + 1
    oW = (X_pad.shape[3] - kW) // S + 1
    
    # 將結果從矩陣形式 Reshape 回 (N, C, H, W) 格式
    Z = Z_row.reshape(N, oH, oW, -1)
    Z = Z.transpose(0, 3, 1, 2) # 調整維度順序為 (Batch, Filter, Height, Width)
    return Z

def conv_backward(dZ, K, oH, oW, kH, kW, S=1, P=0):
    """
    卷積層的反向傳播 (Backward Pass)
    """
    F = dZ.shape[1]  
    # 將上一層傳回的梯度 (dZ) 進行轉置並攤平
    dZ_row = dZ.transpose(0, 2, 3, 1).reshape(-1, F)
    
    # 1. 計算關於卷積核參數的梯度 (dK)
    dK_col = np.dot(X_row.T, dZ_row) 
    dK_col = dK_col.transpose(1, 0)
    dK = dK_col.reshape(K.shape)
    
    # 2. 計算關於偏差項 (Bias) 的梯度 (db)
    db = np.sum(dZ, axis=(0, 2, 3))
    db = db.reshape(-1, F)
    
    # 3. 計算傳給前一層的梯度 (dX)
    K_col = K.reshape(K.shape[0], -1).transpose()  
    dX_row = np.dot(dZ_row, K_col.T)
    
    # 把梯度矩陣還原成原始圖片的形狀
    dX_pad = row2im(dX_row, oH, oW, kH, kW, S)
    
    # 如果當初有做 Padding，現在要把補零的部分切掉
    if P == 0:
        return dX_pad, dK, db
    return dX_pad[:, :, P:-P, P:-P], dK, db
      
def get_im2row_indices(x_shape, kH, kW, S=1, P=0):  
    """
    計算 im2col / im2row 的索引矩陣。
    這是一種向量化 (Vectorized) 優化手段，用來避免 Python 的慢速迴圈。
    """
    N, C, H, W = x_shape
    oH = (H + 2 * P - kH) // S + 1
    oW = (W + 2 * P - kW) // S + 1

    # 計算卷積核內部的相對索引
    i0 = np.repeat(np.arange(kH), kW)
    i0 = np.tile(i0, C)
    # 計算在輸入圖片上的滑動窗口起始位置
    i1 = S * np.repeat(np.arange(oH), oW)
    j0 = np.tile(np.arange(kW), kH * C)
    j1 = S * np.tile(np.arange(oW), oH)
    
    # 組合出最終的座標矩陣
    i = i0.reshape(1, -1) + i1.reshape(-1, 1)
    j = j0.reshape(1, -1) + j1.reshape(-1, 1)
    k = np.repeat(np.arange(C), kH * kW).reshape(1, -1)
    
    return (k, i, j)

def im2row_indices(x, kH, kW, S=1, P=0):  
    """基於索引的高效能 im2row 實作"""
    x_padded = np.pad(x, ((0, 0), (0, 0), (P, P), (P, P)), mode='constant')
    k, i, j = get_im2row_indices(x.shape, kH, kW, S, P)    
    # 透過進階索引 (Advanced Indexing) 一次取出所有資料
    rows = x_padded[:, k, i, j] 
    C = x.shape[1]
    rows = rows.reshape(-1, kH * kW * C) 
    return rows

def row2im_indices(rows, x_shape, kH, kW, S=1, P=0):
    """基於索引的高效 row2im 實作，利用 np.add.at 處理梯度重疊處的累加運算"""
    N, C, H, W = x_shape
    H_pad, W_pad = H + 2 * P, W + 2 * P
    x_pad = np.zeros((N, C, H_pad, W_pad), dtype=rows.dtype)
    k, i, j = get_im2row_indices(x_shape, kH, kW, S, P)
    
    rows_reshaped = rows.reshape(N, -1, C * kH * kW)
 
    # 將攤平的梯度加回對應的像素座標（自動處理多個索引指向同一位置的累加問題）
    np.add.at(x_pad, (slice(None), k, i, j), rows_reshaped)
    
    if P == 0:
        return x_pad
    # 剔除 Padding 部分還原回原始尺寸
    return x_pad[:, :, P:-P, P:-P]