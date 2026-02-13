import numpy as np

class NeuralNetwork:   
    def __init__(self):
        # _layers: 儲存所有層物件（如 Conv, Pool, Dense）的清單
        # _params: 儲存所有可訓練參數（權重與梯度）的引用，方便優化器更新
        self._layers = []
        self._params = []
 
    def add_layer(self, layer):      
        self._layers.append(layer)
        # 如果該層有可訓練參數（如權重 W），則將其參數與對應梯度加入管理清單
        if layer.params: 
            for i, _ in enumerate(layer.params):                         
                self._params.append([layer.params[i], layer.grads[i]])  
                
    def forward(self, X): 
        # 前向傳播：將資料 X 依序流過每一層，每一層的輸出作為下一層的輸入
        for layer in self._layers:            
            X = layer.forward(X) 
        return X   
    
    def __call__(self, X):
        # 語法糖：讓你可以像呼叫函數一樣使用模型，例如 model(X)
        return self.forward(X)
    
    def predict(self, X):
        # 預測模式：取得前向傳播最後的機率分佈，並回傳機率最大的索引（類別）
        p = self.forward(X)       
        if p.ndim == 1:     # 單一樣本處理
            return np.argmax(p)   
        return np.argmax(p, axis=1)
    
    def backward(self, loss_grad, reg = 0.):
        # 反向傳播：從最後一層往第一層傳遞梯度（reversed）
        # loss_grad: 損失函數對輸出的初始梯度
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i] 
            loss_grad = layer.backward(loss_grad)
            # 計算該層參數的正規化梯度（防止過擬合）
            layer.reg_grad(reg) 
        return loss_grad
    
    def reg_loss(self, reg):
        # 正規化損失：加總每一層的權重懲罰項（如 L2 Regularization）
        reg_loss = 0
        for i in range(len(self._layers)):
            reg_loss += self._layers[i].reg_loss(reg)
        return reg_loss
    
    def parameters(self): 
        # 回傳模型中所有「權重與梯度」的配對
        return self._params
    
    def zero_grad(self):
        # 梯度歸零：在每一輪新的訓練開始前，必須將舊的梯度清空，避免累加
        for i, _ in enumerate(self._params):
            self._params[i][1] *= 0.