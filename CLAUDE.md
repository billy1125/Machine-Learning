# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 專案概述

個人機器學習自學與實作練習的筆記庫，包含 Jupyter Notebook 概念說明、從零實作的神經網路（CNN、RNN），以及線性代數與資料結構演算法的學習筆記。

## 環境設定

使用 Conda 管理 Python 環境（Python 3.10）。

```bash
# 建立環境（初次設定）
conda create --name machinelearning python=3.10

# 啟動環境
conda activate machinelearning

# 安裝套件（conda 管理的套件）
conda install jupyter numpy pandas matplotlib scipy scikit-learn statsmodels seaborn

# 安裝 conda 未收錄的套件（透過 pip）
pip install lifelines

# 啟動 Jupyter
jupyter notebook
```

常用 Conda 指令詳見 `env_setting/Conda Prompt.md`。

## 目錄結構

- **根目錄 `*.ipynb`**：主要 ML 主題的 Jupyter Notebook（迴歸、分類、模型評估、正規化、Softmax、Cox 比例風險模型等）
- **`Linear Algebra/`**：線性代數相關 Notebook（行列式、特徵值分解、高斯核、矩陣範數、正定矩陣等）
- **`Data Structures and Algorithms/`**：Python 基礎語法、資料結構（陣列、鏈結串列、堆疊、佇列、樹、圖）、排序與搜尋演算法
- **`python_code/`**：從零實作的神經網路 Python 程式碼
  - `CNN/`：完整 CNN 框架（層定義、訓練器、優化器、權重初始化）
  - `RNN/`：RNN 訓練範例
- **`modules/`**：共用 Python 模組（RNN 相關函式）
- **`Data/`**：訓練用資料集（CSV、MAT 格式）
- **`env_setting/`**：環境設定相關文件

## 從零實作架構（`python_code/`）

這部分是不依賴深度學習框架、純 NumPy 手刻的神經網路：

- **`functionsCnn.py`**：所有層定義（`Layer` 基底類別、`Dense`、`Conv`、`Pool`、`BatchNorm`、`Activation` 等），使用 `from init_weights import *` 載入初始化方法
- **`classCnn.py`**：`NeuralNetwork` 容器類別，管理層串接（`add_layer`）、前向傳播（`forward`/`__call__`）、反向傳播（`backward`）與正規化損失（`reg_loss`）
- **`classTrain.py`**：優化器基底類別 `Optimizer`（含 `zero_grad`/`step`/`regularization`）與子類別 `SGD`、`Adam`、`AdaGrad`、`RMSprop`
- **`init_weights.py`**：參數初始化方法（Kaiming uniform/normal、Xavier 等）
- **`cnnTrainExample.py`**：完整訓練範例

**執行注意事項**：CNN 相關腳本（`cnnTrainExample.py` 等）必須在 `python_code/CNN/` 目錄下執行，才能正確解析本地 import（`from classCnn import NeuralNetwork`、`from functionsCnn import *`）。RNN 範例（`python_code/RNN/rnnTrainSimpleExample.py`）則需從專案根目錄執行，因為 `import modules.rnn as fr` 依賴根目錄的 `modules/` 套件。

## 資料集（`Data/`）

| 檔案 | 用途 |
|------|------|
| `mnist_train.csv` / `mnist_test.csv` | 手寫數字辨識（MNIST） |
| `Iris.csv` | 鳶尾花分類 |
| `heart_failure_clinical_records_dataset.csv` | 心臟衰竭臨床資料（存活分析） |
| `water.csv` / `water.mat` | 水質資料 |
| `Real estate.csv` | 房地產資料（迴歸） |
| `student-por.csv` | 學生成績資料 |
| `food_truck_data.csv` | 餐車銷售資料（簡單迴歸） |

## 使用的套件

| 套件 | 建議版本 | 用途 |
|------|----------|------|
| `numpy` | ≥1.23 | 數值計算（從零實作神經網路的核心） |
| `pandas` | ≥1.5 | 資料載入與處理 |
| `matplotlib` | ≥3.5 | 資料視覺化與訓練曲線繪製 |
| `scipy` | ≥1.9 | 科學計算、統計分析 |
| `scikit-learn` | ≥1.1 | 模型評估、前處理、現成演算法對照 |
| `statsmodels` | ≥0.13 | 統計模型（線性迴歸、GLM） |
| `seaborn` | ≥0.12 | 統計圖表視覺化 |
| `lifelines` | ≥0.27 | 存活分析（Cox 比例風險模型，需 pip 安裝） |
