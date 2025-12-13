-----

# OFDM Channel Estimation Simulation (LML / MMSE / DNN / ELM)

本專案實作了一個 OFDM 系統模擬環境，旨在評估與比較不同的通道估測（Channel Estimation）演算法。專案主要聚焦於重現特定學術論文中的實驗結果（如 Figure 5, 6, 11），比較了傳統統計方法（LS, MMSE）與機器學習方法（Proposed LML, ELM, DNN）在不同信噪比（SNR）與非線性失真環境下的效能。

## ✨ 專案特色

  * **OFDM 系統模擬**：
      * 包含完整的發送端與接收端鏈路（QPSK 調變/解調）。
      * **通道模型**：Pedestrian B 多路徑通道模型。
      * **非線性失真**：實作了 Clipping 失真模型以模擬非線性通道效應。
  * **多種估測器實作**：
      * **LS (Least Square)**：基於導頻（Pilot）的線性插值估測。
      * **MMSE (Minimum Mean Square Error)**：利用通道相關性矩陣的統計估測器。
      * **LML (Proposed Method)**：基於 PATDG (Pilot-Aided Training Data Generation) 的線性機器學習估測器。
      * **ELM (Extreme Learning Machine)**：極限學習機估測器。
      * **DNN (Deep Neural Network)**：基於 PyTorch 的深度神經網路估測器。
  * **效能評估指標**：
      * **NMSE** (Normalized Mean Square Error)
      * **BER** (Bit Error Rate)

## 📂 檔案結構

```
.
├── main.py              # 程式進入點，負責執行模擬並繪製圖表
├── sim.py               # 主要模擬邏輯，定義了針對不同 Figure 的模擬流程
├── OFDMsystem.py        # OFDM 系統類別，包含通道生成、調變、雜訊與 LS 基礎估測
├── LMLestimator.py      # Proposed LML (Linear Machine Learning) 估測器實作
├── MMSE.py              # MMSE 估測器實作
├── Neural_estimators.py # 包含 DNN (PyTorch) 與 ELM 的模型定義
└── README.md            # 專案說明文件
```

## 🛠️ 環境需求

本專案使用 Python 3 開發，並依賴 PyTorch 進行神經網路運算。請確保安裝以下套件：

  * numpy
  * scipy
  * matplotlib
  * torch (建議使用支援 CUDA 的版本以加速運算)

安裝指令範例：

```bash
pip install numpy scipy matplotlib torch
```

## 🚀 使用方式

### 1\. 執行模擬

直接執行 `main.py` 即可開始模擬並產出結果圖表：

```bash
python main.py
```

### 2\. 切換不同的實驗 (Figure)

預設情況下，`main.py` 可能只會執行其中一個實驗。若要執行 Figure 6 或 Figure 11 的模擬，請編輯 `main.py` 檔案底部的 `if __name__ == "__main__":` 區塊，將對應的函數取消註解：

```python
# main.py

if __name__ == "__main__":
    
    # 執行 Figure 5 (NMSE vs SNR)
    plot_fig5()
    
    # 執行 Figure 6 (NMSE vs Dataset Size)
    # plot_fig6()  <-- 取消註解以執行
    
    # 執行 Figure 11 (BER vs Eb/N0, Non-linear)
    # plot_fig11() <-- 取消註解以執行
```

### 3\. 查看結果

執行完成後，結果圖片將會自動儲存於 `./result` 資料夾中：

  * `Figure_5.png`: 不同 SNR 下的 NMSE 比較。
  * `Figure_6.png`: 不同訓練資料集大小對 NMSE 的影響。
  * `Figure_11.png`: 非線性環境下，不同 Eb/N0 的 BER 比較。

## ⚙️ 參數設定

若需調整系統參數（如 FFT 大小、CP 長度、通道參數），可修改 `OFDMsystem.py` 中的 `__init__` 函數：

```python
self.K_total = 512   # Subcarriers total
self.K_active = 410  # Active subcarriers
self.CP = 128        # Cyclic Prefix length
self.Df = 3          # Pilot spacing
```
