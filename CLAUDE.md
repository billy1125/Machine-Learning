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

- **`functionsCnn.py`**：所有層的定義（`Layer` 基底類別、`Dense`、Conv、Pool、BatchNorm、Activation 等）
- **`classCnn.py`**：`NeuralNetwork` 容器類別，管理層的串接、前向/反向傳播、梯度歸零
- **`classTrain.py`**：優化器（`SGD`、`Adam`、`AdaGrad`、`RMSprop`）與訓練迴圈 `train_nn`、資料迭代器 `data_iterator`
- **`init_weights.py`**：參數初始化方法（Kaiming、Xavier 等）
- **`cnnTrainExample.py`**：使用上述模組的完整訓練範例

`modules/` 內的 RNN 模組供 `python_code/RNN/` 的範例引用（`import modules.rnn as fr`）。

## 使用的套件

| 套件 | 用途 |
|------|------|
| `numpy` | 數值計算（矩陣運算、從零實作神經網路的核心） |
| `pandas` | 資料載入與處理 |
| `matplotlib` | 資料視覺化與訓練曲線繪製 |
| `scipy` | 科學計算（統計分析、訊號處理） |
| `sklearn` (scikit-learn) | 機器學習工具（模型評估、前處理、現成演算法對照） |
| `statsmodels` | 統計模型（線性迴歸、GLM） |
| `seaborn` | 統計圖表視覺化 |
| `lifelines` | 存活分析（Cox 比例風險模型） |
| `mpl_toolkits` | 3D 視覺化（matplotlib 延伸） |
