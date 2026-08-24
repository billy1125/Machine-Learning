# 最佳化演算法

本資料夾將「最佳化演算法」拆分為獨立主題教材。每個主題一個 Markdown 檔案，方便分週授課、版本管理與後續擴充。

## 建議閱讀順序

1. [01 最佳化與深度學習](01-optimization-and-deep-learning.md)
2. [02 凸性](02-convexity.md)
3. [03 梯度下降](03-gradient-descent.md)
4. [04 隨機梯度下降](04-stochastic-gradient-descent.md)
5. [05 小批次隨機梯度下降](05-minibatch-sgd.md)
6. [06 動量法](06-momentum.md)
7. [07 AdaGrad](07-adagrad.md)
8. [08 RMSProp](08-rmsprop.md)
9. [09 Adadelta](09-adadelta.md)
10. [10 Adam](10-adam.md)
11. [11 學習率排程器](11-learning-rate-scheduler.md)

## 教材定位

本章的核心不是背誦各種最佳化器公式，而是理解：

```text
Loss
→ Gradient
→ Gradient Descent
→ SGD / Mini-batch
→ Momentum
→ Adaptive Learning Rate
→ Adam
→ Learning Rate Scheduling
```

每一種新方法，都可以視為在修正前一種方法的限制。
