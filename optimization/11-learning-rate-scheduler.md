# 11 學習率排程器

## 學習目標

完成本節後，應能：

- 說明固定學習率的限制。
- 理解訓練初期與後期對步伐大小的需求不同。
- 說明 Step Decay、Exponential Decay 與 Cosine Annealing 的基本概念。
- 理解 Optimizer 與 Learning Rate Scheduler 的分工。

## 1. 為什麼學習率不一定要固定？

訓練初期，模型通常離較好的參數區域很遠。

因此可以：

```text
走比較大的步伐
→ 快速接近較佳區域
```

訓練後期則可能需要：

```text
走比較小的步伐
→ 進行細部調整
```

因此：

> Learning Rate 可以隨訓練過程改變。

## 2. Step Decay

每隔固定 Epoch 降低一次學習率。

例如：

```text
epoch 0–9：   lr = 0.1
epoch 10–19： lr = 0.01
epoch 20–29： lr = 0.001
```

數學形式：

$$
\eta \leftarrow \gamma\eta
$$

其中 $0<\gamma<1$。

## 3. Exponential Decay

學習率逐步下降：

$$
\eta_t
=
\eta_0\gamma^t
$$

其特色是下降較平滑。

## 4. Cosine Annealing

常見形式：

$$
\eta_t
=
\eta_{\min}
+
\frac{1}{2}
(\eta_{\max}-\eta_{\min})
\left(
1+
\cos\frac{\pi t}{T}
\right)
$$

核心直覺是：

> 學習率沿著 Cosine 曲線逐漸下降。

第一次學習不必要求記憶完整公式。

## 5. Optimizer 與 Scheduler 的差別

Optimizer 決定：

> 根據目前梯度，參數怎麼更新。

Scheduler 決定：

> 不同訓練階段，Learning Rate 應該多大。

兩者通常一起使用。

## 6. PyTorch 概念

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=10,
    gamma=0.1
)
```

訓練過程中：

```python
scheduler.step()
```

用來更新學習率。

## 7. 本節核心

可以把 Scheduler 理解成：

```text
Optimizer：
決定往哪裡走，以及如何走

Learning Rate Scheduler：
決定現在應該走多大步
```

這是最佳化演算法從「單一步驟更新」進一步發展到「整個訓練過程管理」的重要概念。
