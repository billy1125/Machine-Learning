# 專案整理建議與規劃草稿

> 本文件為現況分析與改善規劃，尚未實施任何變更。

---

## 一、現況問題盤點

### 1. 嚴重：缺少原始碼檔案

**`modules/rnn.py` 與 `modules/util.py` 遺失**

`modules/__pycache__/` 下存在 `rnn.cpython-310.pyc` 與 `util.cpython-310.pyc`，代表這兩個 `.py` 原始碼曾存在，但目前已消失（可能被意外刪除或未加入版本控制）。`python_code/RNN/rnnTrainSimpleExample.py` 整個依賴 `import modules.rnn as fr`，因此目前 RNN 範例**無法執行**。

**`python_code/CNN/cnnTrainExample.py` 引用不存在的模組**

```python
import functionsAnn as fn    # 檔案不存在
from classAnn import *        # 檔案不存在
```

`functionsAnn`（含 `gen_spiral_dataset`、`cross_entropy_grad_loss`）與 `classAnn` 從未加入版本控制，目前訓練範例同樣**無法執行**。

---

### 2. 中度：Notebook 重複與命名混亂

根目錄存在大量同主題的多版本 Notebook，難以判斷哪份是最新、最完整：

| 主題 | 現有檔案 | 問題 |
|------|----------|------|
| 迴歸 | `Regression with package.ipynb`（16 cells）、`Regression no package.ipynb`（34 cells）、`Regression with package and only analysis.ipynb`（7 cells）、`Regression issues.ipynb` | 4 個版本，定位不清楚 |
| 邏輯迴歸 | `Logistic Regression.ipynb`（11 cells）、`Logistic Regression no package.ipynb`（14 cells）、`Logistic Regression with package.ipynb`（2 cells，幾乎空白）、`Logistic Regression with package and only analysis.ipynb`（11 cells） | 有 2 cells 的空殼未清理 |
| 隨機森林 | `Random Forests.ipynb`（14 cells）、`Random Forests New.ipynb`（24 cells） | 新舊並存 |
| 模型評估 | `Model Evaluation.ipynb`（27 cells）、`Model Evaluation My Study.ipynb`（13 cells） | 用途差異不明確 |
| Softmax | `Softmax no package.ipynb`、`Softmax Regression no package.ipynb`、`Softmax Regression no package MNIST.ipynb` | 3 個版本 |

---

### 3. 輕度：目錄組織問題

- **`Eigen Decomposition.ipynb`** 放在根目錄，但 `Linear Algebra/` 下已有同名檔案 `Eigen Decomposition.ipynb`，構成重複。
- **`AI Learning CLAUDE.md`** 與 **`AI Learning GPT.md`** 是外部課程整理筆記，與程式碼無關，放在根目錄顯得突兀。
- **`Entropy.md`** 與 **`Entropy.ipynb`** 同時存在根目錄，文字筆記與 Notebook 各自記錄同一主題，建議合併。
- **`Data/mnist_train.csv`** 目前為 untracked 狀態（`git status` 顯示 `??`），若需要協作或在其他機器重現，應加入版控或明確加進 `.gitignore` 並附上下載說明。

---

### 4. 輕度：`python_code/` 的套件結構問題

目前 CNN 腳本依賴「切換工作目錄」才能 import，違反 Python 套件慣例：

```
python_code/CNN/
├── classCnn.py
├── classTrain.py
├── functionsCnn.py     # 裡面 from init_weights import *（相對 import）
├── init_weights.py
└── cnnTrainExample.py  # import functionsAnn, classAnn（缺檔）
```

若整個 `python_code/CNN/` 有 `__init__.py` 並從根目錄用 `-m` 執行，就不需要切目錄。

---

## 二、改善建議（依優先順序）

### 優先級 P0：還原缺失原始碼（執行能力修復）

1. **還原 `modules/rnn.py`**：從 `.pyc` 反編譯（`uncompyle6` 或 `decompile3`），或依 RNN 範例的呼叫介面重寫：
   - `fr.rnn_params_init(input_dim, hidden_size, output_dim)`
   - `fr.rnn_hidden_state_init(batch_size, hidden_size)`
   - `fr.rnn_forward(params, Xs, H)` → `(Zs, Hs)`
   - `fr.rnn_backward(params, Xs, Hs, dzs)` → `(dWx, dWh, dbh, dWf, dbf)`
   - `fr.rnn_loss_grad(F, Y, loss_grad_fn, flag)`
   - `fr.mse_loss_grad`
   - `fr.SGD(params, lr, momentum)`

2. **還原或補齊 `functionsAnn.py` 與 `classAnn.py`**：`cnnTrainExample.py` 需要：
   - `fn.gen_spiral_dataset(N, D, K)` — 產生螺旋分類資料
   - `fn.cross_entropy_grad_loss` — 交叉熵損失與梯度
   - `classAnn` — 可能是較早期的 ANN 類別（或已被 `classCnn.py` 取代）

3. 確認還原後，從正確目錄執行兩個訓練範例均能跑通。

---

### 優先級 P1：整理 Notebook 版本

**建議策略：保留最完整版本，其餘刪除或封存**

| 主題 | 建議保留 | 建議處理舊版 |
|------|----------|--------------|
| 迴歸 | `Regression no package.ipynb`（最完整，34 cells，含手刻推導）+ `Regression with package.ipynb`（套件對照） | 刪除 `Regression with package and only analysis.ipynb`；將 `Regression issues.ipynb` 內容併入主檔或改名為 `Regression Debugging Notes.ipynb` |
| 邏輯迴歸 | `Logistic Regression no package.ipynb` + `Logistic Regression with package and only analysis.ipynb` | 刪除 2-cell 空殼的 `Logistic Regression with package.ipynb`；合併 `Logistic Regression.ipynb` 的內容到適合的版本 |
| 隨機森林 | `Random Forests New.ipynb`（24 cells，較新） | 將 `Random Forests.ipynb` 封存或刪除 |
| 模型評估 | `Model Evaluation.ipynb`（27 cells，完整） | 將 `Model Evaluation My Study.ipynb` 的個人筆記部分加入主檔末尾後刪除 |
| Softmax | `Softmax Regression no package MNIST.ipynb`（最完整，用真實資料） | 確認另外兩個的獨特內容後合併或刪除 |

---

### 優先級 P2：目錄重組

**建議新結構：**

```
Machine-Learning/
├── notebooks/                    ← 目前根目錄的 .ipynb 移入（或維持根目錄但整理）
│   ├── regression/
│   ├── classification/
│   └── deep_learning/
├── Linear Algebra/               ← 現有，不動
├── Data Structures and Algorithms/ ← 現有，不動
├── python_code/                  ← 現有，不動（修復 import 問題後）
├── modules/                      ← 補齊 rnn.py / util.py
├── Data/                         ← 現有，補上 mnist_train.csv 的處理說明
├── references/                   ← 新增：放 AI Learning CLAUDE.md / GPT.md
└── env_setting/                  ← 現有，不動
```

> 若移動 Notebook 工作量太大，可改為**只在根目錄新增 `INDEX.md`**，建立學習路徑索引（見下方）。

---

### 優先級 P3：新增學習路徑索引

建立 `INDEX.md` 或在 `README.md` 增加「建議學習順序」區塊：

```
數學基礎
  Linear Algebra/Matrix Basic.ipynb
  Linear Algebra/Determinant.ipynb
  Linear Algebra/Eigen Decomposition.ipynb
  Linear Algebra/Norms.ipynb
  Linear Algebra/Positive Definite.ipynb
  Linear Algebra/Gaussian Kernel.ipynb
  Linear Algebra/Inner(Dot) Product.ipynb
  Log.ipynb → Entropy.ipynb

ML 核心概念
  Regression no package.ipynb         ← 手刻線性迴歸，理解梯度下降
  Regression with package.ipynb       ← sklearn 對照
  Logistic Regression no package.ipynb
  Softmax no package.ipynb
  Regularization.ipynb
  Model Evaluation.ipynb

進階模型
  Batch Normalization.ipynb
  Initial Weight Methods.ipynb
  Common Activation and Loss Functions.ipynb
  Random Forests New.ipynb
  Cox Proportional Hazards Model.ipynb

從零實作（需先讀 ML 核心概念）
  python_code/CNN/cnnTrainExample.py
  python_code/RNN/rnnTrainSimpleExample.py

資料結構與演算法（可獨立學習）
  Data Structures and Algorithms/00~07 + 演算法系列
```

---

### 優先級 P4：`.gitignore` 與版本控制

目前 `Data/mnist_train.csv` 未被追蹤，建議二擇一：

**方案 A — 加入版控**（檔案夠小，< 50 MB 可接受）：
```bash
git add Data/mnist_train.csv
```

**方案 B — 加入 .gitignore 並補充說明**（若檔案過大）：
```gitignore
Data/mnist_train.csv
Data/mnist_test.csv
```
並在 README 補充下載連結（Kaggle MNIST CSV 來源）。

另外建議確認 `.gitignore` 已包含：
```gitignore
__pycache__/
*.pyc
.ipynb_checkpoints/
```

---

## 三、規劃工作清單

| # | 項目 | 優先 | 預估工作量 |
|---|------|------|------------|
| 1 | 還原 `modules/rnn.py`（反編譯或重寫） | P0 | 大 |
| 2 | 還原 `functionsAnn.py` 與 `classAnn.py` | P0 | 中 |
| 3 | 驗證 CNN / RNN 訓練範例可正常執行 | P0 | 小 |
| 4 | 刪除 2-cell 空殼 Notebook | P1 | 小 |
| 5 | 合併 / 刪除重複 Notebook（迴歸、評估、隨機森林） | P1 | 中 |
| 6 | 處理根目錄重複的 `Eigen Decomposition.ipynb` | P2 | 小 |
| 7 | 將 `AI Learning *.md` 移至 `references/` | P2 | 小 |
| 8 | 新增 `INDEX.md` 學習路徑索引 | P3 | 小 |
| 9 | 決定 `mnist_train.csv` 的版控策略並更新 `.gitignore` | P4 | 小 |
| 10 | 補充 `modules/__init__.py`（目前為空） | P4 | 小 |
