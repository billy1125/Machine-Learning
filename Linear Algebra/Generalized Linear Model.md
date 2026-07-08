# 廣義線性模型（GLM）計算原理與數學推導教材

## 1. GLM 是什麼？

廣義線性模型（Generalized Linear Model, GLM）是一種用來分析「自變數如何影響依變數」的統計模型。它可以視為傳統線性迴歸的延伸，特別適合處理依變數不是連續常態分配的情況。

傳統線性迴歸通常假設依變數為連續變數，且誤差項接近常態分配。例如研究「讀書時間」對「考試成績」的影響時，考試成績可以用一般線性迴歸處理。

然而，在許多管理、資訊系統、社會科學與醫學研究中，依變數可能是：

- 是否採用某項系統：是／否
- 使用者是否持續使用 AI 工具：持續／不持續
- 顧客抱怨次數：0 次、1 次、2 次、3 次
- 某事件發生機率：發生／未發生
- 單位時間內的錯誤次數或事故次數

這些資料型態未必符合傳統線性迴歸的常態假設。因此，GLM 透過「機率分配」與「連結函數」將線性模型推廣到更多類型的依變數。

白話來說，GLM 的核心想法是：

> 仍然保留線性模型容易解釋的結構，但允許依變數有不同的資料分配型態。

---

## 2. 傳統線性迴歸的基本形式

在理解 GLM 之前，可以先從傳統線性迴歸開始。

假設研究者想分析使用者的系統滿意度 $Y$，是否受到感知有用性 $X_1$、感知易用性 $X_2$ 與信任 $X_3$ 的影響。

傳統線性迴歸可表示為：

$$
Y_i=\beta_0+\beta_1X_{i1}+\beta_2X_{i2}+\beta_3X_{i3}+\varepsilon_i
$$

其中：

- $Y_i$：第 $i$ 位受測者的依變數數值
- $X_{i1},X_{i2},X_{i3}$：第 $i$ 位受測者在不同自變數上的分數
- $\beta_0$：截距項
- $\beta_1,\beta_2,\beta_3$：迴歸係數
- $\varepsilon_i$：誤差項

若有 $p$ 個自變數，則可寫成一般形式：

$$
Y_i=\beta_0+\beta_1X_{i1}+\beta_2X_{i2}+\cdots+\beta_pX_{ip}+\varepsilon_i
$$

也可以將右側線性組合記為：

$$
\eta_i=\beta_0+\sum_{k=1}^{p}\beta_kX_{ik}
$$

其中，$\eta_i$ 稱為線性預測值（linear predictor）。

在線性迴歸中，模型通常假設：

$$
Y_i\sim N(\mu_i,\sigma^2)
$$

且：

$$
\mu_i=E(Y_i)=\eta_i
$$

也就是說，依變數的期望值 $\mu_i$ 直接等於線性預測值 $\eta_i$。

---

## 3. 為什麼需要 GLM？

傳統線性迴歸雖然容易理解，但它對依變數有較強的假設。若依變數不是連續常態變數，直接使用線性迴歸可能產生不合理結果。

例如，若依變數是「是否持續使用 AI 工具」，可設定：

$$
Y_i=1
$$

表示第 $i$ 位使用者持續使用。

$$
Y_i=0
$$

表示第 $i$ 位使用者未持續使用。

若直接用線性迴歸預測，模型可能得到：

$$
\hat{Y}_i=1.23
$$

或：

$$
\hat{Y}_i=-0.18
$$

但二元變數的合理預測值應該落在 0 到 1 之間，因為它通常代表事件發生的機率。

同樣地，若依變數是「某位使用者一週內遇到系統錯誤的次數」，該變數應為非負整數：

$$
Y_i=0,1,2,3,\ldots
$$

此時若用一般線性迴歸，模型可能預測出負數錯誤次數，這也不具實際意義。

因此，GLM 的目的不是放棄線性模型，而是修正線性模型與依變數之間的關係，使模型更符合不同資料型態。

---

## 4. GLM 的三個核心組成

GLM 主要由三個部分組成：

1. 隨機成分（random component）
2. 系統成分（systematic component）
3. 連結函數（link function）

這三個部分共同決定 GLM 如何描述資料。

---

## 5. 隨機成分：依變數的機率分配

GLM 的第一個核心，是指定依變數 $Y_i$ 服從哪一種機率分配。

在線性迴歸中，常假設：

$$
Y_i\sim N(\mu_i,\sigma^2)
$$

也就是依變數服從常態分配。

但在 GLM 中，依變數可以服從指數族分配（exponential family）中的不同分配，例如：

| 依變數型態 | 常用分配 | 常見模型 |
|---|---|---|
| 連續變數 | Normal distribution | 線性迴歸 |
| 二元變數 | Bernoulli distribution | Logistic regression |
| 成功次數／比例 | Binomial distribution | Logistic regression |
| 計數資料 | Poisson distribution | Poisson regression |
| 過度離散計數資料 | Negative binomial distribution | 負二項迴歸 |
| 正值且右偏連續資料 | Gamma distribution | Gamma regression |

例如，若依變數是使用者是否持續使用系統，可設定：

$$
Y_i\sim Bernoulli(\pi_i)
$$

其中，$\pi_i$ 表示第 $i$ 位使用者持續使用系統的機率。

若依變數是某位使用者一個月內回報錯誤的次數，可設定：

$$
Y_i\sim Poisson(\mu_i)
$$

其中，$\mu_i$ 表示第 $i$ 位使用者在該期間內的期望錯誤次數。

---

## 6. 系統成分：自變數的線性組合

GLM 的第二個核心，是保留線性模型的結構。

不論依變數服從哪一種分配，自變數仍然透過線性組合形成線性預測值：

$$
\eta_i=\beta_0+\beta_1X_{i1}+\beta_2X_{i2}+\cdots+\beta_pX_{ip}
$$

或簡寫為：

$$
\eta_i=\mathbf{x}_i^T\boldsymbol{\beta}
$$

其中：

$$
\mathbf{x}_i=(1,X_{i1},X_{i2},\ldots,X_{ip})^T
$$

$$
\boldsymbol{\beta}=(\beta_0,\beta_1,\beta_2,\ldots,\beta_p)^T
$$

因此：

$$
\mathbf{x}_i^T\boldsymbol{\beta}
=
\beta_0+\beta_1X_{i1}+\beta_2X_{i2}+\cdots+\beta_pX_{ip}
$$

這個部分使 GLM 仍然具有迴歸分析的解釋能力。研究者仍可透過 $\beta_k$ 判斷某個自變數對依變數的影響方向與強度。

---

## 7. 連結函數：把平均數與線性預測值連起來

GLM 的第三個核心，是連結函數（link function）。

在 GLM 中，依變數的期望值記為：

$$
\mu_i=E(Y_i)
$$

但 $\mu_i$ 不一定能直接等於線性預測值 $\eta_i$。因此 GLM 使用連結函數 $g(\cdot)$，將 $\mu_i$ 轉換成線性預測值：

$$
g(\mu_i)=\eta_i
$$

也就是：

$$
g(\mu_i)=\beta_0+\beta_1X_{i1}+\beta_2X_{i2}+\cdots+\beta_pX_{ip}
$$

若要從線性預測值回到依變數的期望值，則使用反連結函數（inverse link function）：

$$
\mu_i=g^{-1}(\eta_i)
$$

白話來說，連結函數的作用是：

> 讓不適合直接用直線表示的依變數平均值，能夠被轉換到適合線性建模的尺度上。

例如，在 logistic regression 中，依變數是 0 或 1，事件機率 $\pi_i$ 必須介於 0 與 1 之間。此時模型不直接讓 $\pi_i$ 等於線性預測值，而是先將機率轉換成 log odds。

---

## 8. GLM 的一般形式

綜合上述三個部分，GLM 的一般形式可表示為：

第一，依變數服從某種分配：

$$
Y_i\sim F(\mu_i,\phi)
$$

其中，$F$ 表示某個指數族分配，$\mu_i$ 為期望值，$\phi$ 為離散參數或尺度參數。

第二，建立線性預測值：

$$
\eta_i=\mathbf{x}_i^T\boldsymbol{\beta}
$$

第三，透過連結函數連接期望值與線性預測值：

$$
g(\mu_i)=\eta_i
$$

因此 GLM 的核心結構可整理為：

$$
g(E(Y_i))=\mathbf{x}_i^T\boldsymbol{\beta}
$$

這個式子是 GLM 最重要的概念。

它表示：

> 不是直接讓依變數等於自變數的線性組合，而是讓依變數的期望值經過連結函數後，等於自變數的線性組合。

---

## 9. Logistic regression：二元依變數的 GLM

Logistic regression 是最常見的 GLM 之一，適合依變數為二元結果的情境。

例如，研究者想分析使用者是否持續使用某 AI 寫作工具：

$$
Y_i=1
$$

表示持續使用。

$$
Y_i=0
$$

表示未持續使用。

此時可設定：

$$
Y_i\sim Bernoulli(\pi_i)
$$

其中：

$$
\pi_i=P(Y_i=1)
$$

表示第 $i$ 位使用者持續使用的機率。

由於 $\pi_i$ 必須介於 0 與 1 之間，不能直接令：

$$
\pi_i=\beta_0+\beta_1X_{i1}+\cdots+\beta_pX_{ip}
$$

否則可能預測出小於 0 或大於 1 的機率。

因此 logistic regression 使用 logit link：

$$
g(\pi_i)=\log\left(\frac{\pi_i}{1-\pi_i}\right)
$$

模型可寫為：

$$
\log\left(\frac{\pi_i}{1-\pi_i}\right)
=
\beta_0+\beta_1X_{i1}+\beta_2X_{i2}+\cdots+\beta_pX_{ip}
$$

其中：

$$
\frac{\pi_i}{1-\pi_i}
$$

稱為 odds，也就是事件發生機率與事件未發生機率的比值。

若將模型轉回機率尺度，可得到：

$$
\pi_i=rac{e^{\eta_i}}{1+e^{\eta_i}}
$$

或：

$$
\pi_i=\frac{1}{1+e^{-\eta_i}}
$$

這個轉換可以保證預測機率一定落在 0 到 1 之間。

---

## 10. Logistic regression 係數的解釋

在 logistic regression 中，$\beta_k$ 不是直接表示依變數增加多少，而是表示 log odds 的變化。

若模型為：

$$
\log\left(\frac{\pi_i}{1-\pi_i}\right)
=
\beta_0+\beta_1X_{i1}
$$

當 $X_{i1}$ 增加 1 單位時，log odds 會增加 $\beta_1$。

若將 $\beta_1$ 取指數，可得到 odds ratio：

$$
OR=e^{\beta_1}
$$

若：

$$
\beta_1=0.7
$$

則：

$$
OR=e^{0.7}\approx 2.01
$$

表示 $X_1$ 每增加 1 單位，事件發生的 odds 約為原本的 2.01 倍。

若：

$$
\beta_1<0
$$

則：

$$
e^{\beta_1}<1
$$

表示 $X_1$ 增加時，事件發生的 odds 下降。

因此，在解釋 logistic regression 時，常會同時報告：

- 迴歸係數 $\beta$
- 標準誤
- Wald test 或 z 值
- p 值
- odds ratio $e^\beta$
- 信賴區間

---

## 11. Poisson regression：計數資料的 GLM

Poisson regression 適合用於分析計數資料，例如：

- 使用者一週內開啟 App 的次數
- 員工一個月內回報系統錯誤的次數
- 工地一天內發生安全違規的次數
- 顧客一段期間內提出抱怨的次數

計數資料通常具有下列特徵：

$$
Y_i=0,1,2,3,\ldots
$$

也就是只能是非負整數。

Poisson regression 假設：

$$
Y_i\sim Poisson(\mu_i)
$$

其中，$\mu_i$ 表示事件發生次數的期望值。

由於 $\mu_i$ 必須大於 0，因此 Poisson regression 常使用 log link：

$$
g(\mu_i)=\log(\mu_i)
$$

模型可寫為：

$$
\log(\mu_i)=\beta_0+\beta_1X_{i1}+\beta_2X_{i2}+\cdots+\beta_pX_{ip}
$$

若轉回原本尺度：

$$
\mu_i=e^{\eta_i}
$$

這可以確保預測的期望次數一定為正數。

---

## 12. Poisson regression 係數的解釋

在 Poisson regression 中，$\beta_k$ 表示自變數對 log expected count 的影響。

若模型為：

$$
\log(\mu_i)=\beta_0+\beta_1X_{i1}
$$

當 $X_{i1}$ 增加 1 單位時，$\log(\mu_i)$ 增加 $\beta_1$。

若將係數取指數，可得到 incident rate ratio，常簡稱 IRR：

$$
IRR=e^{\beta_1}
$$

若：

$$
\beta_1=0.2
$$

則：

$$
IRR=e^{0.2}\approx 1.22
$$

表示 $X_1$ 每增加 1 單位，事件的期望發生次數約變成原本的 1.22 倍，也就是增加約 22%。

若：

$$
\beta_1=-0.3
$$

則：

$$
IRR=e^{-0.3}\approx 0.74
$$

表示 $X_1$ 每增加 1 單位，事件的期望發生次數約變成原本的 0.74 倍，也就是下降約 26%。

---

## 13. GLM 的參數估計：最大概似估計

傳統線性迴歸常可透過最小平方法（ordinary least squares, OLS）估計參數。但 GLM 通常使用最大概似估計（maximum likelihood estimation, MLE）。

最大概似估計的核心想法是：

> 找出一組參數，使目前觀察到的資料出現機率最大。

假設有 $n$ 筆獨立觀察資料：

$$
Y_1,Y_2,\ldots,Y_n
$$

且每一筆資料的機率函數為：

$$
f(y_i;\boldsymbol{\beta})
$$

則整體概似函數為：

$$
L(\boldsymbol{\beta})=\prod_{i=1}^{n}f(y_i;\boldsymbol{\beta})
$$

由於連乘計算較不方便，通常改用對數概似函數：

$$
\ell(\boldsymbol{\beta})=\log L(\boldsymbol{\beta})
$$

因此：

$$
\ell(\boldsymbol{\beta})=
\sum_{i=1}^{n}\log f(y_i;\boldsymbol{\beta})
$$

GLM 的估計目標是找到：

$$
\hat{\boldsymbol{\beta}}
=
\arg\max_{\boldsymbol{\beta}}\ell(\boldsymbol{\beta})
$$

也就是讓對數概似函數最大的參數估計值。

實務上，GLM 的參數通常透過數值最佳化方法求解，例如 Newton-Raphson 或 Fisher scoring。使用統計軟體時，這些計算會由軟體自動完成。

---

## 14. 以使用者是否持續使用 AI 工具為例說明計算邏輯

假設研究者想分析使用者是否持續使用 AI 工具，依變數為：

$$
Y_i=1
$$

表示持續使用。

$$
Y_i=0
$$

表示未持續使用。

自變數包含：

$$
X_1=\text{感知有用性}
$$

$$
X_2=\text{信任}
$$

$$
X_3=\text{使用焦慮}
$$

由於依變數為二元變數，可使用 logistic regression：

$$
Y_i\sim Bernoulli(\pi_i)
$$

其中：

$$
\pi_i=P(Y_i=1)
$$

模型設定為：

$$
\log\left(\frac{\pi_i}{1-\pi_i}\right)
=
\beta_0+\beta_1X_{i1}+\beta_2X_{i2}+\beta_3X_{i3}
$$

假設估計結果為：

$$
\hat{\beta}_0=-3.00
$$

$$
\hat{\beta}_1=0.80
$$

$$
\hat{\beta}_2=0.60
$$

$$
\hat{\beta}_3=-0.50
$$

則模型為：

$$
\log\left(\frac{\pi_i}{1-\pi_i}\right)
=
-3.00+0.80X_{i1}+0.60X_{i2}-0.50X_{i3}
$$

若某位使用者的分數為：

$$
X_{i1}=5
$$

$$
X_{i2}=4
$$

$$
X_{i3}=2
$$

則線性預測值為：

$$
\eta_i=-3.00+0.80(5)+0.60(4)-0.50(2)
$$

$$
\eta_i=-3.00+4.00+2.40-1.00
$$

$$
\eta_i=2.40
$$

接著將 $\eta_i$ 轉換為機率：

$$
\pi_i=\frac{1}{1+e^{-2.40}}
$$

由於：

$$
e^{-2.40}\approx 0.091
$$

所以：

$$
\pi_i\approx \frac{1}{1+0.091}
$$

$$
\pi_i\approx 0.916
$$

表示該使用者持續使用 AI 工具的預測機率約為 91.6%。

---

## 15. 常見連結函數整理

不同 GLM 會搭配不同的連結函數。常見整理如下：

| 依變數型態 | 分配 | 常用連結函數 | 模型形式 |
|---|---|---|---|
| 連續變數 | Normal | Identity link | $\mu_i=\eta_i$ |
| 二元變數 | Bernoulli / Binomial | Logit link | $\log\left(\frac{\pi_i}{1-\pi_i}\right)=\eta_i$ |
| 計數資料 | Poisson | Log link | $\log(\mu_i)=\eta_i$ |
| 正值右偏資料 | Gamma | Log link 或 inverse link | $\log(\mu_i)=\eta_i$ |
| 比例資料 | Binomial | Logit link | $\log\left(\frac{\pi_i}{1-\pi_i}\right)=\eta_i$ |

需要注意的是，分配與連結函數的選擇應根據研究問題、依變數型態與資料特性決定，而不是只根據研究者習慣選擇。

---

## 16. GLM 的模型檢查

建立 GLM 後，研究者需要檢查模型是否適合資料。常見檢查包括：

### 16.1 係數顯著性

可檢查每個自變數的係數是否顯著不等於 0。常見檢定包括：

- Wald test
- Likelihood ratio test
- Score test

若某個自變數的係數顯著，表示該變數與依變數之間存在統計上可檢測的關聯。

### 16.2 模型整體適配度

GLM 常使用 deviance 評估模型適配情形。Deviance 可理解為模型與飽和模型之間的差距。

一般來說，deviance 越小，表示模型與資料越接近。

也可使用資訊準則比較模型：

$$
AIC=-2\ell(\hat{\boldsymbol{\beta}})+2k
$$

其中：

- $\ell(\hat{\boldsymbol{\beta}})$：模型的最大對數概似值
- $k$：模型參數數量

AIC 越小，通常表示模型在解釋力與複雜度之間取得較佳平衡。

### 16.3 過度離散

在 Poisson regression 中，一個重要假設是：

$$
E(Y_i)=Var(Y_i)=\mu_i
$$

也就是平均數與變異數相等。

但實際資料中，計數變數常出現變異數大於平均數的情況，稱為過度離散（overdispersion）：

$$
Var(Y_i)>E(Y_i)
$$

若有明顯過度離散，Poisson regression 可能低估標準誤，導致顯著性判斷過於樂觀。此時可考慮：

- quasi-Poisson model
- negative binomial regression
- robust standard errors

---

## 17. GLM 的整體流程整理

GLM 的分析流程可整理如下：

1. 確認研究問題與依變數型態。
2. 判斷依變數適合的機率分配，例如常態、二元、二項、Poisson 或 Gamma。
3. 選擇合適的連結函數，例如 identity、logit 或 log link。
4. 建立線性預測式：

$$
\eta_i=\mathbf{x}_i^T\boldsymbol{\beta}
$$

5. 建立 GLM 形式：

$$
g(E(Y_i))=\mathbf{x}_i^T\boldsymbol{\beta}
$$

6. 使用最大概似估計求得參數。
7. 檢查係數方向、顯著性與效果量。
8. 檢查模型適配度，例如 deviance、AIC 或殘差診斷。
9. 根據模型類型解釋結果，例如 odds ratio 或 incident rate ratio。
10. 回到研究問題，說明統計結果的理論與實務意義。

---

## 18. GLM 的研究應用意義

在資訊系統與管理研究中，GLM 適合用來處理許多非連續常態依變數。例如，若研究者想分析使用者是否採用某項 AI 系統，依變數通常是二元變數，此時 logistic regression 會比一般線性迴歸更合適。

若研究者關心的是某段期間內的使用次數、錯誤次數、投訴次數或事件發生次數，則可考慮 Poisson regression 或 negative binomial regression。這類模型能保留迴歸分析的解釋性，同時更符合計數資料的統計特性。

GLM 的價值在於，它提供了一個統一框架，讓研究者可以根據依變數的資料型態選擇合適模型，而不是把所有研究問題都硬套進傳統線性迴歸。對於實證研究而言，GLM 能提升模型設定的合理性，也能讓研究結論更貼近資料本身的結構。

若再結合資訊系統研究中的理論架構，例如 Technology Acceptance Model、IS Continuance Model、Trust in AI 或 Human-AI Collaboration，GLM 可用來檢驗不同心理變數、系統特徵與使用情境如何影響使用者是否採用、是否持續使用、以及使用行為發生的頻率。
