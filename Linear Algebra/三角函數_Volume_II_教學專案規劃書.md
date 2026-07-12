# 三角函數 Volume II：從單位圓到圖形語言

## 一、專案概述

本教學專案將三角函數重新整理成一條由「單位圓」出發的理解主線，避免學生只將三角函數視為需要記憶的零散公式。

教學內容將以 Markdown 文件呈現概念、公式、圖形解釋與練習，並搭配一個可直接在瀏覽器執行的 HTML 互動動畫。學生可透過滑桿改變角度，同步觀察：

- 單位圓上的點如何移動。
- 點的 \(x\) 座標如何對應 \(\cos x\)。
- 點的 \(y\) 座標如何對應 \(\sin x\)。
- \(\tan x\)、\(\cot x\)、\(\sec x\)、\(\csc x\) 的數值如何變化。
- 函數圖上的對應位置如何隨角度移動。
- 函數無定義時，垂直漸近線如何出現。

---

## 二、專案目標

### 2.1 教學目標

完成本單元後，學習者應能：

1. 從單位圓理解 \(\sin x\) 與 \(\cos x\)。
2. 說明單位圓上的點為何可寫成：

   \[
   P(\cos x,\sin x)
   \]

3. 理解 sine graph 與 cosine graph 如何由單位圓座標形成。
4. 認識三角函數的振幅、週期、定義域和值域。
5. 從比值關係理解 \(\tan x\) 與 \(\cot x\)。
6. 從倒數關係理解 \(\sec x\) 與 \(\csc x\)。
7. 判斷各三角函數的無定義點與垂直漸近線。
8. 使用互動動畫觀察角度、座標、函數值與圖形之間的同步變化。

### 2.2 技術目標

專案預計產出：

- 一份主要 Markdown 教學文件。
- 一份可獨立執行的 HTML 互動動畫。
- 可重複使用的 JavaScript 繪圖與計算邏輯。
- 可擴充至 GitHub Pages、教學網站或簡報嵌入的靜態教材。

---

## 三、預期產出

建議的專案結構如下：

```text
trigonometry-volume-ii/
├─ README.md
├─ docs/
│  ├─ 01-unit-circle.md
│  ├─ 02-sine-cosine.md
│  ├─ 03-tangent-cotangent.md
│  ├─ 04-secant-cosecant.md
│  ├─ 05-symmetry-periodicity.md
│  └─ exercises.md
├─ interactive/
│  ├─ index.html
│  ├─ style.css
│  └─ app.js
├─ assets/
│  ├─ images/
│  └─ diagrams/
└─ LICENSE
```

若希望降低初期複雜度，也可先採用單檔形式：

```text
trigonometry-volume-ii/
├─ 三角函數_Volume_II_教學文件.md
└─ 三角函數_單位圓互動動畫.html
```

---

## 四、教學核心主線

整體教學依照下列關係展開：

```text
單位圓
  ↓
點 P(cos x, sin x)
  ↓
x 座標與 y 座標
  ↓
cosine graph 與 sine graph
  ↓
比值函數 tan、cot
  ↓
倒數函數 sec、csc
  ↓
定義域、值域、週期、對稱與漸近線
```

核心觀點是：

> 三角函數描述的是角度在圓上移動時，座標、比值與倒數如何變化。

---

## 五、Markdown 教學文件規劃

## 5.1 第一章：從單位圓理解三角函數

### 教學內容

介紹半徑為 1、圓心在原點的單位圓。

當角度 \(x\) 從正 \(x\) 軸開始旋轉時，圓上的點可表示為：

\[
P(\cos x,\sin x)
\]

其中：

- \(\cos x\) 是點的 \(x\) 座標。
- \(\sin x\) 是點的 \(y\) 座標。

### 教學重點

- 角度可使用 degree 或 radian 表示。
- 單位圓半徑固定為 1。
- 座標值必須位於 \([-1,1]\)。
- 不同象限會影響正負號。

### 建議圖像

- 單位圓。
- 旋轉角度。
- 水平投影 \(\cos x\)。
- 垂直投影 \(\sin x\)。
- 點 \(P(\cos x,\sin x)\)。

---

## 5.2 第二章：sine graph 與 cosine graph

當角度持續增加時，單位圓上的點會繞圓旋轉。

- 點的 \(y\) 座標形成 \(y=\sin x\)。
- 點的 \(x\) 座標形成 \(y=\cos x\)。

### 共同性質

| 性質 | \(y=\sin x\) | \(y=\cos x\) |
|---|---:|---:|
| 振幅 | 1 | 1 |
| 週期 | \(2\pi\) | \(2\pi\) |
| 定義域 | \(\mathbb{R}\) | \(\mathbb{R}\) |
| 值域 | \([-1,1]\) | \([-1,1]\) |

### 關鍵角度

| \(x\) | \(\sin x\) | \(\cos x\) |
|---:|---:|---:|
| \(0\) | 0 | 1 |
| \(\frac{\pi}{2}\) | 1 | 0 |
| \(\pi\) | 0 | -1 |
| \(\frac{3\pi}{2}\) | -1 | 0 |
| \(2\pi\) | 0 | 1 |

### 延伸觀念

\[
\sin(-x)=-\sin x
\]

因此 sine 是奇函數。

\[
\cos(-x)=\cos x
\]

因此 cosine 是偶函數。

---

## 5.3 第三章：tangent 與 cotangent

### 定義

\[
\tan x=\frac{\sin x}{\cos x}
\]

\[
\cot x=\frac{\cos x}{\sin x}
\]

### tangent 的無定義點

當：

\[
\cos x=0
\]

分母為 0，因此：

\[
x=\frac{\pi}{2}+k\pi,\quad k\in\mathbb{Z}
\]

在這些位置，\(y=\tan x\) 會出現垂直漸近線。

### cotangent 的無定義點

當：

\[
\sin x=0
\]

分母為 0，因此：

\[
x=k\pi,\quad k\in\mathbb{Z}
\]

在這些位置，\(y=\cot x\) 會出現垂直漸近線。

### 共同觀察

- \(\tan x\) 的週期為 \(\pi\)。
- \(\cot x\) 的週期為 \(\pi\)。
- 兩者值域皆為 \(\mathbb{R}\)。
- 函數圖會被垂直漸近線分成多個區段。

---

## 5.4 第四章：secant 與 cosecant

### 定義

\[
\sec x=\frac{1}{\cos x}
\]

\[
\csc x=\frac{1}{\sin x}
\]

### 值域

由於：

\[
-1\leq \sin x\leq 1
\]

以及：

\[
-1\leq \cos x\leq 1
\]

取倒數後，\(\sec x\) 與 \(\csc x\) 不會落在 \(-1\) 與 \(1\) 之間。

因此值域為：

\[
(-\infty,-1]\cup[1,\infty)
\]

### 圖形特徵

- 圖形由多段 U 形或倒 U 形分支構成。
- \(\sec x\) 在 \(\cos x=0\) 的位置出現垂直漸近線。
- \(\csc x\) 在 \(\sin x=0\) 的位置出現垂直漸近線。
- 每個分支會接觸 \(y=1\) 或 \(y=-1\)。

---

## 5.5 第五章：將觀念整合

本單元需整合以下概念：

| 中文 | English | 核心問題 |
|---|---|---|
| 對稱 | symmetry | 函數圖在原點或 \(y\) 軸兩側有何規律？ |
| 週期 | periodicity | 圖形經過多少角度會重複？ |
| 化簡 | reduction formulas | 如何把較大或負角度轉成已知角度？ |
| 函數圖 | graphs | 函數值如何隨角度變化？ |
| 定義域 | domain | 哪些角度可以代入？ |
| 值域 | range | 函數值可能落在哪些範圍？ |
| 漸近線 | asymptote | 函數在哪些位置快速趨向無限大？ |

---

## 六、HTML 互動動畫規劃

## 6.1 互動介面

HTML 頁面至少包含：

1. 角度滑桿。
2. 角度數值顯示。
3. 單位圓 Canvas。
4. sine graph Canvas。
5. cosine graph Canvas。
6. 即時函數值表格。
7. 特殊角度快速按鈕。
8. 動畫播放與暫停按鈕。
9. 函數無定義警示。
10. 角度制與弧度制切換。

---

## 6.2 主要互動流程

使用者拖曳角度滑桿後：

```text
讀取角度
  ↓
換算弧度
  ↓
計算 sin、cos、tan、cot、sec、csc
  ↓
更新單位圓上的點
  ↓
更新投影線
  ↓
更新函數圖上的對應點
  ↓
更新數值表格
  ↓
判斷是否接近無定義點
  ↓
顯示警示或垂直漸近線說明
```

---

## 6.3 即時計算項目

JavaScript 需要計算：

```javascript
const sinValue = Math.sin(angle);
const cosValue = Math.cos(angle);
const tanValue = sinValue / cosValue;
const cotValue = cosValue / sinValue;
const secValue = 1 / cosValue;
const cscValue = 1 / sinValue;
```

需使用容許誤差判斷分母是否接近 0：

```javascript
const EPSILON = 1e-8;

const tanDefined = Math.abs(cosValue) > EPSILON;
const cotDefined = Math.abs(sinValue) > EPSILON;
```

---

## 6.4 單位圓動畫

單位圓需顯示：

- 圓心 \(O\)。
- 半徑 1。
- 旋轉角度。
- 點 \(P(\cos x,\sin x)\)。
- 水平投影。
- 垂直投影。
- \(\cos x\) 標籤。
- \(\sin x\) 標籤。

畫布座標轉換：

```javascript
screenX = centerX + radius * Math.cos(angle);
screenY = centerY - radius * Math.sin(angle);
```

Canvas 的 \(y\) 軸向下，因此需要使用負號修正。

---

## 6.5 函數圖動畫

建議初版先繪製：

- \(y=\sin x\)
- \(y=\cos x\)

後續版本再加入：

- \(y=\tan x\)
- \(y=\cot x\)
- \(y=\sec x\)
- \(y=\csc x\)

函數圖中應顯示：

- \(x\) 軸與 \(y\) 軸。
- \(0\)、\(\frac{\pi}{2}\)、\(\pi\)、\(\frac{3\pi}{2}\)、\(2\pi\)。
- 完整曲線。
- 目前角度的垂直參考線。
- 目前函數值的標記點。

---

## 6.6 播放動畫

動畫模式會自動增加角度：

```javascript
angle += speed;
```

到達 \(2\pi\) 後重新回到 0：

```javascript
if (angle > 2 * Math.PI) {
  angle = 0;
}
```

建議提供：

- 播放。
- 暫停。
- 重設。
- 速度調整。
- 單次循環。
- 連續循環。

---

## 七、教學流程建議

### 第一階段：單位圓與座標

教師先顯示單位圓，讓學生觀察點在不同角度的位置。

建議角度：

- \(0\)
- \(\frac{\pi}{6}\)
- \(\frac{\pi}{4}\)
- \(\frac{\pi}{3}\)
- \(\frac{\pi}{2}\)

### 第二階段：座標形成函數圖

拖曳滑桿，觀察：

- \(x\) 座標如何形成 cosine graph。
- \(y\) 座標如何形成 sine graph。

### 第三階段：比值函數

使用即時數值說明：

\[
\tan x=\frac{\sin x}{\cos x}
\]

當 \(\cos x\) 越接近 0，\(\tan x\) 的絕對值越大。

### 第四階段：倒數函數

使用即時數值說明：

\[
\sec x=\frac{1}{\cos x}
\]

當 \(|\cos x|<1\) 時：

\[
|\sec x|>1
\]

### 第五階段：整合觀察

讓學生同時觀察：

- 角度。
- 座標。
- 函數值。
- 函數圖。
- 定義域。
- 值域。
- 漸近線。

---

## 八、練習設計

### 基礎題

1. 當 \(x=\frac{\pi}{3}\) 時，求 \(\sin x\) 與 \(\cos x\)。
2. 當 \(x=\pi\) 時，單位圓上的點位於哪一個位置？
3. \(y=\sin x\) 的週期是多少？
4. \(y=\cos x\) 的值域是多少？

### 觀察題

1. 哪些角度會讓 \(\tan x\) 無定義？
2. 哪些角度會讓 \(\csc x\) 無定義？
3. 當 \(\cos x\) 接近 0 時，\(\sec x\) 如何變化？
4. \(\sin x\) 與 \(\cos x\) 的圖形有何平移關係？

### 解釋題

1. 為什麼 \(\sec x\) 的值不會落在 \((-1,1)\)？
2. 為什麼 \(\tan x\) 的週期是 \(\pi\)？
3. 為什麼三角函數可以視為一套圖形語言？

---

## 九、開發階段

### 階段一：Markdown 教學文件

- 完成單位圓章節。
- 完成 sine 與 cosine 章節。
- 完成 tangent 與 cotangent 章節。
- 完成 secant 與 cosecant 章節。
- 加入公式、表格與練習。

### 階段二：基礎 HTML 動畫

- 建立角度滑桿。
- 繪製單位圓。
- 顯示即時座標。
- 繪製 sine graph。
- 繪製 cosine graph。

### 階段三：進階函數

- 加入 tangent。
- 加入 cotangent。
- 加入 secant。
- 加入 cosecant。
- 加入無定義提示。
- 加入垂直漸近線。

### 階段四：教學優化

- 加入特殊角度按鈕。
- 加入播放與暫停。
- 加入動畫速度控制。
- 加入角度制與弧度制切換。
- 改善手機與平板版面。

---

## 十、驗收標準

### Markdown 文件

- 公式顯示正確。
- 章節順序清楚。
- 圖形與文字概念一致。
- 每章包含例子與練習。
- 中英文術語一致。

### HTML 動畫

- 滑桿可正常改變角度。
- 單位圓上的點位置正確。
- \(\sin x\) 與 \(\cos x\) 數值正確。
- 函數圖標記點與角度同步。
- 無定義點顯示清楚。
- 動畫可播放、暫停與重設。
- 桌面與行動裝置皆可正常使用。

---

## 十一、後續擴充

完成基礎版本後，可加入：

- 多圈旋轉，例如 \(-2\pi\) 到 \(4\pi\)。
- 角度制與弧度制雙向顯示。
- 三角函數恆等式視覺化。
- 相位平移。
- 振幅調整。
- 頻率調整。
- \(y=A\sin(Bx+C)+D\) 互動圖形。
- 練習題自動產生。
- 學習紀錄與答題回饋。
- GitHub Pages 線上發布。

---

## 十二、小結

本專案不是將三角函數拆成六個獨立公式，而是建立一條統一的理解主線：

> 單位圓上的角度移動，會產生座標；座標形成 sine 與 cosine；座標的比值形成 tangent 與 cotangent；座標的倒數形成 secant 與 cosecant。

透過 Markdown 文件建立完整概念架構，再透過 HTML 動畫將角度、座標、函數值與函數圖同步呈現，可以讓三角函數從抽象公式轉變為可觀察、可操作的圖形語言。
