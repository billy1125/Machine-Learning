# 05 小批次隨機梯度下降

## 學習目標

完成本節後，應能：

- 說明 Mini-batch 的目的。
- 區分 Batch、SGD 與 Mini-batch SGD。
- 理解 Batch Size、Epoch 與 Iteration。
- 說明 Mini-batch 為何適合 GPU。

## 1. Mini-batch 的核心想法

若每次只用一筆資料，梯度太不穩定。

若每次使用全部資料，又太慢。

因此折衷做法是：

> 每次使用一小批資料。

例如：

```text
Dataset：100,000 筆

每次訓練：
32 筆
64 筆
128 筆
256 筆
```

## 2. Mini-batch 梯度

假設一個 Batch 為：

$$
B=\{i_1,i_2,\ldots,i_m\}
$$

則：

$$
g_t=
\frac{1}{m}
\sum_{i\in B}
\nabla L_i(\theta_t)
$$

更新：

$$
\theta_{t+1}=\theta_t-\eta g_t
$$

## 3. 三種方法比較

| 方法 | 每次使用資料量 | 梯度穩定性 | 計算特性 |
|---|---:|---|---|
| Batch GD | 全部 | 高 | 大型資料集較慢 |
| SGD | 1 筆 | 低 | 單次更新快 |
| Mini-batch SGD | 一小批 | 中高 | 適合 GPU |

## 4. Epoch、Batch、Iteration

假設：

```text
Dataset size = 10,000
Batch size = 100
```

則一個 Epoch 有：

$$
\frac{10000}{100}=100
$$

次 Iteration。

### Epoch

全部訓練資料完整被使用一次。

### Batch

一次送進模型的資料集合。

### Iteration

完成一次參數更新。

## 5. 為什麼 GPU 喜歡 Mini-batch？

GPU 擅長大量平行矩陣運算。

例如一次計算：

```text
128 筆資料 × 多個神經元
```

通常比一筆一筆處理更能利用硬體。

因此 Mini-batch 同時兼顧：

- 梯度穩定度。
- 運算效率。
- GPU 平行化。

## 6. Batch Size 的影響

Batch Size 太小：

- 梯度 Noise 較大。
- 更新次數多。

Batch Size 太大：

- 記憶體需求增加。
- 每個 Epoch 的更新次數變少。

實務上必須依資料、模型與硬體調整。
