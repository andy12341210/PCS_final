-----

# OFDM Channel Estimation Simulation (LML / MMSE / DNN / ELM)

本專案實作了一個 OFDM 系統模擬環境，旨在重現學術論文中的實驗結果，並評估與比較不同通道估測（Channel Estimation）演算法的效能。

專案比較了傳統統計方法（LS, MMSE）與基於機器學習的方法（Proposed LML, ELM, DNN）在不同環境下的表現。

## ✨ 專案內容

* **OFDM 系統模擬**：
    * 完整的發送與接收鏈路（QPSK 調變/解調）。
    * **通道模型**：Pedestrian B 多路徑衰減通道。
    * **非線性失真**：實作 Clipping 模型以模擬非線性效應。
    * **STO (Symbol Timing Offset)**：模擬符號定時偏移造成的相位旋轉與符號間干擾 (ISI)。
    * **CFO (Carrier Frequency Offset)**：模擬載波頻率偏移造成的子載波間干擾 (ICI)。
* **多種估測器實作**：
    * **LS (Least Square)**：基於導頻（Pilot）的線性插值基礎基準。
    * **MMSE (Minimum Mean Square Error)**：利用通道相關性矩陣的統計估測器（最佳理論值）。
    * **LML (Proposed Method)**：
        * **PATDG**：僅利用導頻數據進行訓練的低複雜度線性學習器。
        * **DDTDG**：利用判決回授（Decision-Directed）進行在線更新的版本。
    * **Deep Learning**：
        * **DNN**：基於 PyTorch 的全連接神經網路 (ChannelNet)。
        * **ELM (Extreme Learning Machine)**：複數域極限學習機，具備快速訓練特性。

## 📂 專案結構

⚠️ **注意**：根據程式碼引用邏輯，請確保您的資料夾結構如下所示（請建立一個名為 `alg` 的資料夾並將估測器檔案放入）：

```text
Project_Root/
├── alg/                     # 演算法模組資料夾
│   ├── __init__.py          # (可選，若無此檔 Python 3.3+ 亦可執行)
│   ├── LMLestimator.py      # LML 估測器核心
│   ├── MMSE.py              # MMSE 估測器核心
│   └── Neural_estimators.py # DNN 與 ELM 模型 (PyTorch)
├── main.py                  # 程式進入點 (負責繪圖與呼叫模擬)
├── sim.py                   # 模擬流程控制 (針對不同 Figure 的迴圈邏輯)
├── OFDMsystem.py            # OFDM 系統參數與物理層模擬
├── README.md                # 說明文件
└── requirements.txt         # 虛擬環境所需套件
```

## 🛠️ 環境安裝

本專案使用 Python 進行 OFDM 系統模擬與通道估測演算法比較。為了確保所有程式碼（包含 PyTorch 深度學習模型）能順利運作，請依照以下步驟建置環境。

建立虛擬環境 (Virtual Environment)
我們建議使用虛擬環境，以避免與系統其他的 Python 套件產生衝突。以下是 Conda 的安裝方式

```bash
# 1. 建立 Python 3.8 環境 (環境名稱設為 ofdm_env)
conda create -n ofdm_env python=3.10.0

# 2. 啟動環境
conda activate ofdm_env

# 3. 安裝核心套件 (Numpy, Scipy, Matplotlib) 與 PyTorch
#    (Conda 會自動處理 PyTorch 與 CUDA 的相容性)
conda install -r requirements.txt
```

## 🚀 使用方式

### 1\. 執行模擬

直接執行 `main.py` 即可開始模擬並產出結果圖表：

```bash
python main.py
```

### 2\. 切換不同的實驗 (Figure)

預設情況下，`main.py` 會執行所有實驗。

```python
# main.py

if __name__ == "__main__":
    
    # 執行 Figure 5 (NMSE vs SNR)
    plot_fig5()
    
    # 執行 Figure 6 (NMSE vs Dataset Size)
    plot_fig6()

    # 執行 Figure 8/9...
```

### 3\. 查看結果

執行完成後，結果圖片將會自動儲存於 `./result` 資料夾中：

  * `Figure_5.png`: 不同 SNR 下 NMSE/BER 的比較。
  * `Figure_6.png`: 不同訓練資料集大小對 NMSE 的影響。
  * `Figure_8_9.png`: STO/CFO 對 NMSE/BER 的影響。
  * `Figure_11.png`: 非線性環境下，不同 Eb/N0 下 BER 的比較。

## ⚙️ 參數設定

若需調整系統參數（如 FFT 大小、CP 長度、通道參數），可修改 `OFDMsystem.py` 中的 `__init__` 函數：

```python
self.K_total = 512   # Subcarriers total
self.K_active = 410  # Active subcarriers
self.CP = 128        # Cyclic Prefix length
self.Df = 3          # Pilot spacing
```
