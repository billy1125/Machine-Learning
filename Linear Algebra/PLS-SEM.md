# PLS-SEM 計算原理與數學推導教材

## 1. PLS-SEM 是什麼？

偏最小平方法結構方程模型（Partial Least Squares Structural Equation Modeling, PLS-SEM）是一種以變異解釋與預測為導向的結構方程模型方法。

SEM（Structural Equation Modeling）主要用來處理潛在變數之間的關係。例如研究者可能想知道：

- 系統品質是否會影響使用者滿意度？
- 信任是否會影響持續使用意圖？
- 感知有用性是否會透過態度進一步影響採用意圖？
- AI 解釋性是否會提升信任，進而影響人機協作績效？

這些概念通常無法直接觀察，因此稱為潛在變數（latent variables）。研究者通常會透過多個問卷題項來測量這些概念。

PLS-SEM 的核心想法是：

> 透過指標資料估計潛在變數分數，再用這些潛在變數分數估計變數之間的路徑關係。

相較於共變異數導向 SEM（Covariance-Based SEM, CB-SEM），PLS-SEM 更重視解釋變異與預測能力，對樣本數與資料分配的要求通常較彈性，因此常用於探索性研究、預測導向研究、模型較複雜或樣本數相對有限的研究情境。

---

## 2. PLS-SEM 的基本組成

PLS-SEM 通常包含兩個部分：

| 模型部分 | 英文名稱 | 主要目的 |
|---|---|---|
| 測量模型 | Measurement model / Outer model | 說明潛在變數如何由觀察指標測量 |
| 結構模型 | Structural model / Inner model | 說明潛在變數之間的因果或理論關係 |

以資訊系統持續使用研究為例，研究者可能設定：

- 系統品質（System Quality）
- 使用者滿意度（User Satisfaction）
- 持續使用意圖（Continuance Intention）

其中，每個潛在變數都由多個題項測量。例如：

$$
\text{System Quality}=\{SQ_1,SQ_2,SQ_3\}
$$

$$
\text{User Satisfaction}=\{SAT_1,SAT_2,SAT_3\}
$$

$$
\text{Continuance Intention}=\{CI_1,CI_2,CI_3\}
$$

結構模型可能假設：

$$
\text{System Quality}\rightarrow \text{User Satisfaction}
$$

$$
\text{User Satisfaction}\rightarrow \text{Continuance Intention}
$$

意思是：系統品質會影響使用者滿意度，而使用者滿意度會進一步影響持續使用意圖。

---

## 3. 潛在變數與觀察指標

PLS-SEM 中的核心概念是潛在變數。潛在變數無法直接觀察，只能透過觀察指標進行測量。

假設有一個潛在變數：

$$
\xi
$$

其觀察指標為：

$$
x_1,x_2,x_3
$$

則可表示為：

$$
\xi \leftarrow \{x_1,x_2,x_3\}
$$

例如，「信任」本身無法被直接看到，但可以透過問卷題項測量：

- 我相信此系統提供的建議是可靠的。
- 我相信此系統會依照我的利益運作。
- 我相信此系統的輸出結果是可信的。

這些題項就是用來反映或形成「信任」這個潛在變數的觀察指標。

---

## 4. 反映式與形成式測量模型

PLS-SEM 中，測量模型常分為兩種型態：

1. 反映式測量模型（Reflective measurement model）
2. 形成式測量模型（Formative measurement model）

這兩者的差異非常重要，因為它會影響模型估計、信效度檢驗與結果解釋。

---

## 5. 反映式測量模型

反映式測量模型的邏輯是：

> 潛在變數導致觀察指標的變化。

也就是說，潛在變數是原因，題項是結果。

可表示為：

$$
x_j=\lambda_j \xi+\varepsilon_j
$$

其中：

- $x_j$：第 $j$ 個觀察指標
- $\xi$：潛在變數
- $\lambda_j$：外部負荷量（outer loading）
- $\varepsilon_j$：測量誤差

例如，若一個人對系統高度信任，則他在多個信任題項上的分數通常都會偏高。

反映式模型的特徵包括：

- 指標之間通常高度相關。
- 刪除某個題項通常不會改變構念的核心意義。
- 主要檢查 loading、Cronbach's alpha、Composite Reliability、AVE、HTMT 等信效度指標。

例如：

$$
\text{Trust}\rightarrow T_1,T_2,T_3
$$

其中，Trust 是潛在變數，$T_1,T_2,T_3$ 是由信任反映出來的題項。

---

## 6. 形成式測量模型

形成式測量模型的邏輯是：

> 觀察指標共同形成潛在變數。

也就是說，題項是原因，潛在變數是結果。

可表示為：

$$
\xi=w_1x_1+w_2x_2+\cdots+w_px_p
$$

其中：

- $x_j$：第 $j$ 個觀察指標
- $w_j$：外部權重（outer weight）
- $\xi$：由指標加權形成的潛在變數

例如，「社經地位」可能由教育程度、收入、職業聲望共同形成。這些指標不一定高度相關，但每個指標都可能代表構念的一個重要面向。

形成式模型的特徵包括：

- 指標之間不一定高度相關。
- 刪除某個指標可能改變構念意義。
- 主要檢查共線性、outer weight 顯著性與內容效度。
- 不適合用 Cronbach's alpha 或 AVE 評估內部一致性。

例如：

$$
\text{Socioeconomic Status}\leftarrow \{\text{Education},\text{Income},\text{Occupation}\}
$$

意思是教育、收入與職業共同形成社經地位。

---

## 7. 結構模型的基本形式

結構模型描述潛在變數之間的關係。

假設有三個潛在變數：

$$
\eta_1,\eta_2,\eta_3
$$

若研究假設為：

$$
\eta_1\rightarrow \eta_2
$$

$$
\eta_2\rightarrow \eta_3
$$

則可以寫成：

$$
\eta_2=\beta_{21}\eta_1+\zeta_2
$$

$$
\eta_3=\beta_{32}\eta_2+\zeta_3
$$

其中：

- $\beta_{21}$：$\eta_1$ 對 $\eta_2$ 的路徑係數
- $\beta_{32}$：$\eta_2$ 對 $\eta_3$ 的路徑係數
- $\zeta_2,\zeta_3$：結構模型殘差

若套用在資訊系統情境：

$$
\text{Satisfaction}=\beta_1\text{System Quality}+\zeta_1
$$

$$
\text{Continuance Intention}=\beta_2\text{Satisfaction}+\zeta_2
$$

則 $\beta_1$ 與 $\beta_2$ 就是研究者關心的主要路徑係數。

---

## 8. PLS-SEM 的資料矩陣表示

假設研究中有 $n$ 位受測者與 $p$ 個觀察指標，資料矩陣可表示為：

$$
X=
\begin{bmatrix}
x_{11} & x_{12} & \cdots & x_{1p}\\
x_{21} & x_{22} & \cdots & x_{2p}\\
\vdots & \vdots & \ddots & \vdots\\
x_{n1} & x_{n2} & \cdots & x_{np}
\end{bmatrix}
$$

其中：

- $n$：樣本數
- $p$：觀察指標數
- $x_{ij}$：第 $i$ 位受測者在第 $j$ 個指標上的分數

若某潛在變數 $\xi_k$ 由 $p_k$ 個指標測量，則可取出對應的指標矩陣：

$$
X_k=
\begin{bmatrix}
x_{11}^{(k)} & x_{12}^{(k)} & \cdots & x_{1p_k}^{(k)}\\
x_{21}^{(k)} & x_{22}^{(k)} & \cdots & x_{2p_k}^{(k)}\\
\vdots & \vdots & \ddots & \vdots\\
x_{n1}^{(k)} & x_{n2}^{(k)} & \cdots & x_{np_k}^{(k)}
\end{bmatrix}
$$

PLS-SEM 會根據每一個潛在變數的指標矩陣，估計該潛在變數的分數。

---

## 9. 潛在變數分數的估計

PLS-SEM 的一個重要特徵是會估計每位受測者在每個潛在變數上的分數。

假設潛在變數 $\xi_k$ 有 $p_k$ 個指標，則其分數可表示為：

$$
\hat{\xi}_k=X_k w_k
$$

其中：

- $X_k$：潛在變數 $\xi_k$ 的觀察指標矩陣
- $w_k$：外部權重向量
- $\hat{\xi}_k$：估計出的潛在變數分數

若 $\xi_k$ 有三個指標：

$$
X_k=(x_1,x_2,x_3)
$$

則：

$$
\hat{\xi}_k=w_1x_1+w_2x_2+w_3x_3
$$

簡單來說，PLS-SEM 會用題項的加權組合來代表潛在變數。

---

## 10. PLS-SEM 的演算法概念

PLS-SEM 的估計通常透過迭代演算法完成。簡化後，可分成下列步驟：

1. 初始化每個潛在變數的外部權重。
2. 根據外部權重計算潛在變數分數。
3. 根據結構模型中潛在變數之間的關係，更新內部估計。
4. 根據內部估計重新更新外部權重。
5. 重複步驟 2 至 4，直到權重變化足夠小。
6. 使用最後的潛在變數分數估計結構模型路徑係數。

PLS-SEM 的核心不是一次求出封閉解，而是透過反覆更新，使潛在變數分數逐漸穩定。

---

## 11. 外部模型估計：Mode A 與 Mode B

PLS-SEM 常見的外部權重更新方式包含 Mode A 與 Mode B。

### 11.1 Mode A：常用於反映式模型

Mode A 通常用於反映式測量模型。其概念是用每個指標與潛在變數內部估計值的相關來更新權重。

若潛在變數的內部估計為：

$$
z_k
$$

則第 $j$ 個指標的權重可近似表示為：

$$
w_{kj}\propto \text{cor}(x_{kj},z_k)
$$

也就是說，與潛在變數內部估計越相關的指標，其權重越高。

### 11.2 Mode B：常用於形成式模型

Mode B 通常用於形成式測量模型。其概念是用多元迴歸方式估計權重。

若潛在變數內部估計為：

$$
z_k
$$

且指標矩陣為：

$$
X_k
$$

則可透過迴歸估計：

$$
z_k=X_k w_k+e_k
$$

得到外部權重：

$$
w_k=(X_k^TX_k)^{-1}X_k^Tz_k
$$

因此，Mode B 比較接近「哪些指標可以共同形成或預測此潛在變數」的邏輯。

---

## 12. 內部模型估計

內部模型估計是指根據潛在變數之間的結構關係，更新每個潛在變數的內部估計值。

假設有三個潛在變數：

$$
\xi_1,\xi_2,\xi_3
$$

其中：

$$
\xi_1\rightarrow \xi_2
$$

$$
\xi_2\rightarrow \xi_3
$$

在 PLS-SEM 演算法中，會根據相鄰潛在變數的分數建立內部估計。例如：

$$
z_2=e_{21}\hat{\xi}_1+e_{23}\hat{\xi}_3
$$

其中 $e_{21}$ 與 $e_{23}$ 可依照不同 weighting scheme 計算，例如 centroid scheme、factorial scheme 或 path weighting scheme。

簡化理解即可：內部估計會根據模型中與該潛在變數相連的其他潛在變數分數進行更新。

---

## 13. 路徑係數估計

當潛在變數分數穩定後，PLS-SEM 會使用普通最小平方法（Ordinary Least Squares, OLS）估計結構模型中的路徑係數。

假設結構模型為：

$$
\eta= \beta_1\xi_1+\beta_2\xi_2+\zeta
$$

則可寫成矩陣形式：

$$
\eta=X_\xi\beta+\zeta
$$

其中：

$$
X_\xi=
\begin{bmatrix}
\xi_{11} & \xi_{12}\\
\xi_{21} & \xi_{22}\\
\vdots & \vdots\\
\xi_{n1} & \xi_{n2}
\end{bmatrix}
$$

OLS 估計式為：

$$
\hat{\beta}=(X_\xi^TX_\xi)^{-1}X_\xi^T\eta
$$

因此，PLS-SEM 中的結構路徑係數可以理解為：先估計潛在變數分數，再用這些分數進行迴歸分析。

---

## 14. 反映式測量模型評估

若測量模型為反映式，通常需要檢查下列項目：

1. Indicator reliability
2. Internal consistency reliability
3. Convergent validity
4. Discriminant validity

---

## 15. 外部負荷量與指標信度

外部負荷量（outer loading）表示觀察指標與潛在變數之間的關聯強度。

若反映式模型為：

$$
x_j=\lambda_j\xi+\varepsilon_j
$$

則 $\lambda_j$ 即為外部負荷量。

常見判斷方式是：

$$
|\lambda_j| \geq 0.70
$$

通常表示指標具有不錯的解釋力。

因為：

$$
0.70^2=0.49
$$

表示潛在變數約能解釋該指標接近一半的變異。

若 loading 太低，研究者需要檢查該題項是否不適合測量該構念。不過是否刪題不能只看數值，也要考慮理論與內容效度。

---

## 16. Cronbach's Alpha

Cronbach's alpha 是傳統上常用的內部一致性指標。

假設某構念有 $k$ 個題項，題項變異數為 $\sigma_i^2$，總分變異數為 $\sigma_T^2$，則：

$$
\alpha=\frac{k}{k-1}\left(1-\frac{\sum_{i=1}^{k}\sigma_i^2}{\sigma_T^2}\right)
$$

其中：

- $k$：題項數
- $\sigma_i^2$：第 $i$ 個題項的變異數
- $\sigma_T^2$：所有題項加總後的總分變異數

$\alpha$ 越高，代表題項之間內部一致性越高。

但在 PLS-SEM 中，Cronbach's alpha 常被視為較保守的信度估計，因為它假設所有題項 loading 相同。實務上通常也會報告 Composite Reliability。

---

## 17. Composite Reliability

Composite Reliability（CR）是 PLS-SEM 常用的內部一致性指標。

若某構念的標準化外部負荷量為：

$$
\lambda_1,\lambda_2,\ldots,\lambda_k
$$

則 Composite Reliability 可表示為：

$$
CR=\frac{(\sum_{i=1}^{k}\lambda_i)^2}{(\sum_{i=1}^{k}\lambda_i)^2+\sum_{i=1}^{k}(1-\lambda_i^2)}
$$

其中：

$$
1-\lambda_i^2
$$

可理解為第 $i$ 個題項的誤差變異。

一般而言，Composite Reliability 高於 0.70 常被視為可接受，但若過高，例如高於 0.95，可能表示題項過度重複。

---

## 18. Average Variance Extracted

平均變異萃取量（Average Variance Extracted, AVE）用來評估收斂效度。

其計算方式為：

$$
AVE=\frac{\sum_{i=1}^{k}\lambda_i^2}{k}
$$

其中：

- $\lambda_i$：第 $i$ 個題項的標準化外部負荷量
- $k$：題項數

一般常用判斷標準為：

$$
AVE\geq 0.50
$$

表示潛在變數平均能解釋其指標超過一半的變異。

---

## 19. 區辨效度：Fornell-Larcker 準則

Fornell-Larcker 準則用來檢查不同構念是否彼此具有足夠區辨性。

其基本概念是：某構念的 AVE 平方根應大於該構念與其他構念的相關係數。

若第 $k$ 個構念的 AVE 為：

$$
AVE_k
$$

則需檢查：

$$
\sqrt{AVE_k}>r_{kq}, \quad q\neq k
$$

其中：

- $r_{kq}$：構念 $k$ 與構念 $q$ 的相關係數

若一個構念與其他構念的相關過高，甚至高於自身 AVE 平方根，表示這些構念可能不容易區分。

---

## 20. 區辨效度：HTMT

HTMT（Heterotrait-Monotrait Ratio）是近年 PLS-SEM 中常用的區辨效度檢查方法。

HTMT 的概念是比較：

1. 不同構念之間題項相關的平均值
2. 同一構念內題項相關的平均值

簡化表示為：

$$
HTMT_{ab}=
\frac{
\text{mean}(|r_{ij}|),\ i\in a,\ j\in b
}{
\sqrt{
\text{mean}(|r_{ij}|),\ i,j\in a
\times
\text{mean}(|r_{ij}|),\ i,j\in b
}
}
$$

常見判斷方式包括：

$$
HTMT<0.85
$$

或：

$$
HTMT<0.90
$$

具體門檻需依研究領域、構念相近程度與文獻慣例判斷。

---

## 21. 形成式測量模型評估

若測量模型為形成式，評估邏輯與反映式不同。

形成式模型常見檢查項目包括：

1. 內容效度
2. 指標共線性
3. outer weight 顯著性
4. outer loading 的輔助判斷

形成式模型的指標共同形成構念，因此不應用內部一致性指標來判斷。也就是說，不應用 Cronbach's alpha、Composite Reliability 或 AVE 來評估形成式構念。

---

## 22. 形成式模型的共線性檢查

形成式模型中，各指標共同預測或形成潛在變數，因此需要檢查指標之間是否存在嚴重共線性。

常用指標為 Variance Inflation Factor（VIF）。

若第 $j$ 個指標被其他指標迴歸後的決定係數為：

$$
R_j^2
$$

則 VIF 為：

$$
VIF_j=\frac{1}{1-R_j^2}
$$

若 VIF 過高，表示該指標與其他指標高度重疊，可能造成 outer weight 不穩定。

常見判斷方式是：

$$
VIF<5
$$

較嚴格研究也可能採用：

$$
VIF<3.3
$$

---

## 23. 結構模型評估

結構模型評估主要關注潛在變數之間的路徑關係是否符合研究假設。

常見評估項目包括：

1. 共線性檢查
2. 路徑係數
3. 決定係數 $R^2$
4. 效果量 $f^2$
5. 預測相關性 $Q^2$
6. 模型預測能力

---

## 24. 結構路徑係數

路徑係數表示一個潛在變數對另一個潛在變數的影響方向與強度。

假設模型為：

$$
\eta=\beta_1\xi_1+\beta_2\xi_2+\zeta
$$

其中：

- $\beta_1$：$\xi_1$ 對 $\eta$ 的影響
- $\beta_2$：$\xi_2$ 對 $\eta$ 的影響

若：

$$
\beta_1>0
$$

表示 $\xi_1$ 對 $\eta$ 有正向影響。

若：

$$
\beta_1<0
$$

表示 $\xi_1$ 對 $\eta$ 有負向影響。

但路徑係數是否具有統計意義，通常需要透過 bootstrapping 檢定。

---

## 25. 決定係數 $R^2$

決定係數 $R^2$ 表示外生潛在變數對內生潛在變數變異的解釋比例。

假設某內生變數為 $\eta$，其預測值為 $\hat{\eta}$，則：

$$
R^2=1-\frac{\sum_{i=1}^{n}(\eta_i-\hat{\eta}_i)^2}{\sum_{i=1}^{n}(\eta_i-\bar{\eta})^2}
$$

其中：

- $\eta_i$：實際潛在變數分數
- $\hat{\eta}_i$：模型預測值
- $\bar{\eta}$：潛在變數平均值

若：

$$
R^2=0.60
$$

表示模型中的前因變數可以解釋該內生變數 60% 的變異。

在 PLS-SEM 中，$R^2$ 是非常重要的指標，因為 PLS-SEM 本身較偏向解釋變異與預測導向。

---

## 26. 效果量 $f^2$

效果量 $f^2$ 用來評估某個外生變數對內生變數 $R^2$ 的貢獻。

其計算方式為：

$$
f^2=\frac{R^2_{\text{included}}-R^2_{\text{excluded}}}{1-R^2_{\text{included}}}
$$

其中：

- $R^2_{\text{included}}$：包含某外生變數時的 $R^2$
- $R^2_{\text{excluded}}$：移除某外生變數後的 $R^2$

若移除某個變數後 $R^2$ 明顯下降，表示該變數對模型解釋力有重要貢獻。

常見參考標準為：

| $f^2$ | 解釋 |
|---:|---|
| 0.02 | 小效果 |
| 0.15 | 中效果 |
| 0.35 | 大效果 |

這些標準只是參考，仍需結合理論與研究脈絡判斷。

---

## 27. Bootstrapping 檢定

PLS-SEM 通常不直接依賴常態分配假設來估計路徑係數標準誤，而是使用 bootstrapping。

Bootstrapping 的基本流程如下：

1. 從原始樣本中進行有放回抽樣。
2. 每次抽出與原始樣本相同大小的 bootstrap sample。
3. 對每個 bootstrap sample 重新估計 PLS-SEM 模型。
4. 重複多次，例如 5,000 次。
5. 根據 bootstrap 分布計算標準誤、t 值、p 值與信賴區間。

若原始路徑係數為：

$$
\hat{\beta}
$$

bootstrap 標準誤為：

$$
SE(\hat{\beta})
$$

則 t 值可表示為：

$$
t=\frac{\hat{\beta}}{SE(\hat{\beta})}
$$

若信賴區間不包含 0，通常表示該路徑係數達統計顯著。

---

## 28. 中介效果分析

PLS-SEM 常用於檢驗中介效果。

假設模型為：

$$
X\rightarrow M\rightarrow Y
$$

其中：

- $X$：自變數
- $M$：中介變數
- $Y$：依變數

路徑係數為：

$$
X\rightarrow M=a
$$

$$
M\rightarrow Y=b
$$

則間接效果為：

$$
a\times b
$$

若：

$$
a\times b
$$

的 bootstrap 信賴區間不包含 0，通常表示中介效果成立。

例如：

$$
\text{AI Explainability}\rightarrow \text{Trust}\rightarrow \text{Usage Intention}
$$

其中，Trust 可能是 AI 解釋性影響使用意圖的中介機制。

---

## 29. 調節效果分析

調節效果是指某個變數會改變兩個變數之間的關係強度。

假設模型為：

$$
Y=\beta_1X+\beta_2Z+\beta_3(XZ)+\zeta
$$

其中：

- $X$：主要自變數
- $Z$：調節變數
- $XZ$：交互作用項
- $\beta_3$：調節效果

若 $\beta_3$ 顯著，表示 $Z$ 會改變 $X$ 對 $Y$ 的影響。

例如：

$$
\text{AI Explainability}\times \text{AI Literacy}\rightarrow \text{Trust}
$$

表示 AI 素養可能會調節 AI 解釋性對信任的影響。

---

## 30. PLS-SEM 的整體分析流程

PLS-SEM 的研究流程可整理如下：

1. 建立理論模型與研究假設。
2. 定義潛在變數與測量題項。
3. 判斷每個構念是反映式或形成式。
4. 收集問卷或觀察資料。
5. 資料清理與描述統計。
6. 估計 PLS-SEM 模型。
7. 評估反映式或形成式測量模型。
8. 評估結構模型。
9. 使用 bootstrapping 檢定路徑係數。
10. 檢驗中介、調節或多群組差異。
11. 報告結果並回應研究假設。

---

## 31. PLS-SEM 與 CB-SEM 的差異

PLS-SEM 與 CB-SEM 都屬於 SEM 方法，但研究取向不同。

| 比較項目 | PLS-SEM | CB-SEM |
|---|---|---|
| 主要目標 | 解釋變異、預測 | 理論檢定、模型配適 |
| 估計基礎 | 變異導向 | 共變異數導向 |
| 潛在變數分數 | 會明確估計 | 通常不是主要輸出 |
| 樣本與分配要求 | 較具彈性 | 通常較嚴格 |
| 模型配適 | 較不以整體配適為核心 | 重視整體模型配適 |
| 適用情境 | 探索性、預測導向、複雜模型 | 驗證性理論檢定、成熟模型 |

需要注意的是，PLS-SEM 並不是 CB-SEM 的簡化替代品。兩者適合不同研究目的。若研究重點是檢驗成熟理論模型的整體配適，CB-SEM 可能較合適；若研究重點是預測、解釋變異、探索新模型或處理形成式構念，PLS-SEM 可能較合適。

---

## 32. PLS-SEM 的研究應用情境

PLS-SEM 常見於資訊系統、管理、行銷、服務科學、人因工程與社會科學研究。

例如，在 MIS 領域，可以用於研究：

1. 生成式 AI 使用意圖
2. AI 信任與解釋性
3. 人機協作績效
4. 資訊系統持續使用
5. 數位平台採用
6. 智慧穿戴裝置接受度
7. 線上學習系統滿意度
8. 資訊安全遵循意圖

以 AI 系統研究為例，可能建立如下模型：

$$
\text{Explainability}\rightarrow \text{Trust}
$$

$$
\text{Trust}\rightarrow \text{Usage Intention}
$$

$$
\text{Perceived Usefulness}\rightarrow \text{Usage Intention}
$$

$$
\text{AI Literacy}\times \text{Explainability}\rightarrow \text{Trust}
$$

這樣的模型可以同時處理直接效果、中介效果與調節效果。

---

## 33. PLS-SEM 結果報告建議

PLS-SEM 結果報告通常可依下列順序呈現：

### 33.1 樣本與資料說明

包含：

- 樣本數
- 樣本來源
- 問卷尺度
- 缺失值處理
- 常態性或描述統計
- 共同方法偏誤檢查，如研究需要

### 33.2 測量模型結果

反映式模型可報告：

- outer loading
- Cronbach's alpha
- Composite Reliability
- AVE
- Fornell-Larcker
- HTMT

形成式模型可報告：

- VIF
- outer weight
- outer loading
- bootstrapping 顯著性
- 內容效度說明

### 33.3 結構模型結果

可報告：

- path coefficient
- t 值
- p 值
- confidence interval
- $R^2$
- $f^2$
- $Q^2$ 或預測能力
- 中介效果與調節效果

### 33.4 假設檢定表

常見表格格式如下：

| 假設 | 路徑 | 係數 | t 值 | p 值 | 結論 |
|---|---|---:|---:|---:|---|
| H1 | System Quality → Satisfaction | 0.42 | 5.31 | < .001 | 支持 |
| H2 | Satisfaction → Continuance Intention | 0.58 | 7.44 | < .001 | 支持 |

---

## 34. 常見錯誤與注意事項

使用 PLS-SEM 時，常見問題包括：

1. 未區分反映式與形成式構念。
2. 對形成式構念使用 Cronbach's alpha 或 AVE。
3. 只報告路徑係數，不報告測量模型品質。
4. 只看 p 值，不解釋效果量與理論意義。
5. 未檢查共線性。
6. 中介或調節效果未使用 bootstrapping。
7. 將 PLS-SEM 誤解為小樣本時一定適用。
8. 在理論未清楚時過度堆疊路徑。
9. 未說明刪題理由。
10. 將預測導向方法用於完全驗證性模型，卻不說明方法選擇理由。

PLS-SEM 雖然在樣本與分配上較有彈性，但不代表可以忽略研究設計、測量品質與理論基礎。

---

## 35. 本教材範例總結

以資訊系統持續使用為例，PLS-SEM 可用來檢驗：

$$
\text{System Quality}\rightarrow \text{Satisfaction}
$$

$$
\text{Satisfaction}\rightarrow \text{Continuance Intention}
$$

分析流程為：

1. 用多個題項測量每個潛在變數。
2. 判斷題項是反映式或形成式。
3. 估計潛在變數分數。
4. 檢查測量模型信效度。
5. 估計結構路徑係數。
6. 使用 bootstrapping 檢查顯著性。
7. 報告 $R^2$、$f^2$ 與假設檢定結果。

---

## 36. 核心觀念整理

PLS-SEM 的數學精神在於：用觀察指標估計潛在變數分數，再用這些潛在變數分數估計理論模型中的路徑關係。

簡單來說，PLS-SEM 不是單純的迴歸，也不是只看問卷題項平均數，而是結合測量模型與結構模型的一套完整分析程序。

若研究者關心的是複雜模型中的預測關係、潛在變數之間的路徑、中介或調節效果，且模型中包含多個不可直接觀察的構念，PLS-SEM 是相當實用的方法。不過，使用時仍必須清楚說明為何選擇 PLS-SEM，而不是只因為樣本數較小或軟體容易操作就採用。
