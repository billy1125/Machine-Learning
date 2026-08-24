# 09 Adadelta

## 學習目標

完成本節後，應能：

- 說明 Adadelta 與 AdaGrad 的歷史關係。
- 理解 Adadelta 不只考慮梯度，也考慮參數更新尺度。
- 知道 Adadelta 在現代深度學習教材中的定位。

## 1. 為什麼會出現 Adadelta？

AdaGrad 的主要問題是：

> 有效學習率會隨時間持續下降。

Adadelta 嘗試進一步改善這個問題。

## 2. 基本想法

Adadelta 同時追蹤：

1. 梯度平方的移動平均。
2. 過去參數更新量平方的移動平均。

因此它不只是根據梯度大小調整步伐，也參考過去實際更新的尺度。

## 3. 教學定位

第一次學習最佳化演算法時，不需要要求學生完整背誦 Adadelta 的全部公式。

較重要的是理解演進：

```text
AdaGrad
   ↓
有效學習率越來越小
   ↓
RMSProp / Adadelta
   ↓
改用近期資訊調整更新尺度
```

## 4. 為什麼仍值得學？

雖然目前實務上 Adam 更常作為入門重點，但 Adadelta 可以幫助理解：

> 最佳化器的發展，是持續針對前一代方法的限制進行修正。

## 5. 延伸方向

進階教材可再加入：

- 梯度平方的 Exponential Moving Average。
- 更新量平方的 Exponential Moving Average。
- 單位尺度的直覺。
- 與 RMSProp 的比較。
