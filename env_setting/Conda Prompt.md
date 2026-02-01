# 常用的 Conda 指令

指令參考自以下網站

https://ithelp.ithome.com.tw/articles/10255411

## 虛擬環境管理

### 建立虛擬環境

這邊先假設環境名稱為test123，並且指定python版次為3.8，環境名稱及python版次可以依自己需求而異。

`conda create --name test123 python=3.8`

### 切換到特定虛擬環境

`conda activate test123`

### 離開虛擬環境

`conda deactivate`

### 列出所有虛擬環境

`conda env list`

### 清除所有暫存檔(ex.套件安裝檔)

`conda clean --all`

### 刪除虛擬環境

`conda env remove --name test123`

### 複製虛擬環境

`conda create --name test1234 --clone test123`

## 套件處理

### 安裝套件

如果你要指定特定版本，可在套件名稱後面加上「==」與版本號。

`conda install numpy==X.XX.X`

### 刪除環境中的套件

`conda uninstall numpy`

### 列出虛擬環境下已安裝的所有套件

要先切換到特定虛擬環境

`conda list`

## 備份

備份當前環境，並且把備份檔命名為test48763，備份檔名稱可以依照自己需求設定，如果你不指定名稱，預設的備份檔儲存路徑為C:\Windows\System32

`conda env export > test48763.yaml`

如果要指定備份檔儲存路徑，則：

`conda env export > C:\Users\ZongSing_NB2\Documents\test48763.yaml`

## 還原

匯入備份檔test48763.yaml，並以此為基礎建立新環境test1314520，同樣的，預設的備份檔讀取路徑為C:\Windows\System32

`conda env create --file test48763.yaml --name test1314520`

若備份檔test48763.yaml在其他地方，則需指定讀取路徑。

`conda env create --file C:\Users\ZongSing_NB2\Documents\test48763.yaml --name test1314520`