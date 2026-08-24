# 08 RMSProp

## 學習目標

完成本節後，應能：

- 說明 RMSProp 如何改善 AdaGrad。
- 理解梯度平方的指數移動平均。
- 解釋 RMSProp 的自適應學習率概念。

## 1. AdaGrad 的問題

AdaGrad 累積：

$$
s_t=s_{t-1}+g_t^2
$$

因此歷史資訊永遠保留。

問題是：

```text
累積值越來越大
→ 有效學習率越來越小
→ 後期幾乎不再更新
```

## 2. RMSProp 的核心想法

RMSProp 不永久保存所有歷史梯度，而是讓：

> 最近的梯度比較重要，很久以前的梯度逐漸淡化。

## 3. 指數移動平均

$$
s_t
=
\rho s_{t-1}
+
(1-\rho)g_t^2
$$

其中：

- $\rho$：保留舊資訊的比例。
- $g_t^2$：目前梯度平方。

## 4. 更新公式

$$
\theta_{t+1}
=
\theta_t
-
\frac{\eta}
{\sqrt{s_t+\epsilon}}
g_t
$$

## 5. 與 AdaGrad 的差異

AdaGrad：

```text
永遠累積全部歷史
```

RMSProp：

```text
保留近期歷史
逐漸忘記很久以前的梯度
```

因此 RMSProp 可以避免有效學習率持續衰減到極小。

## 6. 與 Adam 的關係

RMSProp 主要追蹤：

> 梯度平方的移動平均。

Adam 之後會把這個概念與 Momentum 結合。
