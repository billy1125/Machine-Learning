# 10 Adam

## 學習目標

完成本節後，應能：

- 說明 Adam 的全名與核心概念。
- 理解 Adam 如何結合 Momentum 與 RMSProp。
- 區分 First Moment 與 Second Moment。
- 理解 Bias Correction 的目的。
- 閱讀 PyTorch 中 Adam 的基本設定。

## 1. Adam 是怎麼來的？

Adam 的全名為：

> Adaptive Moment Estimation

可以把 Adam 理解成：

```text
Momentum
   +
RMSProp
   ↓
Adam
```

Momentum 主要記錄：

> 最近梯度的方向。

RMSProp 主要記錄：

> 最近梯度平方的尺度。

## 2. First Moment

Adam 計算：

$$
m_t
=
\beta_1m_{t-1}
+
(1-\beta_1)g_t
$$

可以把 $m_t$ 理解為：

> 最近梯度方向的平滑平均。

## 3. Second Moment

Adam 同時計算：

$$
v_t
=
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2
$$

可以把 $v_t$ 理解為：

> 最近梯度大小平方的平滑平均。

## 4. Bias Correction

一開始：

$$
m_0=0
$$

$$
v_0=0
$$

因此早期估計容易偏向 0。

Adam 使用：

$$
\hat{m}_t
=
\frac{m_t}{1-\beta_1^t}
$$

$$
\hat{v}_t
=
\frac{v_t}{1-\beta_2^t}
$$

進行修正。

## 5. Adam 更新

完整更新可寫成：

$$
\theta_{t+1}
=
\theta_t
-
\eta
\frac{\hat{m}_t}
{\sqrt{\hat{v}_t}+\epsilon}
$$

其中：

- $\hat{m}_t$：控制主要方向。
- $\hat{v}_t$：控制不同參數的有效步伐。

## 6. PyTorch

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)
```

這一行背後其實包含：

```text
Gradient
+
Moving Average
+
Squared Gradient
+
Adaptive Learning Rate
+
Bias Correction
```

## 7. Adam 並不是永遠最好

Adam 常被使用，是因為：

- 對許多問題容易得到合理結果。
- 對不同參數自動調整有效學習率。
- 通常對初學者較容易訓練。

但不同模型與任務仍可能適合：

- SGD。
- SGD + Momentum。
- AdamW。
- 其他進階最佳化器。

因此不應把 Adam 理解成「所有任務的固定答案」。
