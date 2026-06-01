# 人工智慧演算法與實作練習

個人學習機器學習與深度學習的筆記庫，涵蓋理論概念的 Jupyter Notebook、從零實作的神經網路，以及相關的線性代數與資料結構筆記。

## 內容架構

- **機器學習模型**：迴歸、邏輯迴歸、Softmax、隨機森林、Cox 比例風險模型、模型評估
- **深度學習**：CNN 與 RNN 的純 NumPy 從零實作
- **數學基礎**：線性代數（特徵值分解、高斯核、正定矩陣、行列式）、熵、對數
- **資料結構與演算法**：陣列、鏈結串列、堆疊、佇列、樹、圖、排序與搜尋

## 環境需求

使用 **Conda** 管理虛擬環境（Python 3.10）。

```bash
conda activate machinelearning
jupyter notebook
```

## 使用套件

| 套件 | 版本建議 | 用途 |
|------|----------|------|
| `numpy` | ≥1.23 | 數值計算（從零實作神經網路的核心） |
| `pandas` | ≥1.5 | 資料載入與處理 |
| `matplotlib` | ≥3.5 | 資料視覺化 |
| `scipy` | ≥1.9 | 科學計算、統計分析 |
| `scikit-learn` | ≥1.1 | 機器學習工具、模型評估、前處理 |
| `statsmodels` | ≥0.13 | 統計模型（線性迴歸、GLM） |
| `seaborn` | ≥0.12 | 統計圖表視覺化 |
| `lifelines` | ≥0.27 | 存活分析（Cox 比例風險模型） |

安裝：

```bash
conda install numpy pandas matplotlib scipy scikit-learn statsmodels seaborn
pip install lifelines
```

## 資料集

`Data/` 目錄下包含練習用的資料集：

- `mnist_train.csv` / `mnist_test.csv`：手寫數字辨識
- `Iris.csv`：鳶尾花分類
- `heart_failure_clinical_records_dataset.csv`：心臟衰竭臨床資料
- `water.csv` / `water.mat`：水質資料
- `Real estate.csv`：房地產資料
- `student-por.csv`：學生成績資料
- `food_truck_data.csv`：餐車銷售資料
