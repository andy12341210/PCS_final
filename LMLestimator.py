import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from OFDMsystem import OFDMSystem

# 假設已經定義了 OFDMSystem (如上一段對話所示)
# 如果沒有，請確保引入上一段的 OFDMSystem 類別

class LML_Estimator:
    def __init__(self, M, pilot_spacing, test):
        """
        LML 信道估計器
        :param M: 每個 Group 的導頻數量 (論文通常設 M=2 或 3)
        :param pilot_spacing: 導頻間隔 Df
        """
        self.M = M
        self.Df = pilot_spacing
        
        # 計算一個 Group 覆蓋的子載波長度
        # 例如 M=2, Df=4 -> Pilot(0)...Data...Pilot(4), 長度為 5
        self.group_len = (M - 1) * pilot_spacing + 1
        
        # 定義窗口內的相對索引
        # 導頻在窗口內的相對位置: 0, Df, 2Df...
        self.pilot_rel_idx = np.arange(0, self.group_len, pilot_spacing)
        
        # 數據在窗口內的相對位置: 排除導頻位置
        all_rel_idx = np.arange(self.group_len)
        self.data_rel_idx = np.setdiff1d(all_rel_idx, self.pilot_rel_idx)
        
        self.S = len(self.data_rel_idx) # 數據子載波數量
        self.W_d = None # 權重矩陣

        self.test = test
        
        if test:
            print(f"LML 模型初始化: M={M}, Df={pilot_spacing}, Group Length={self.group_len}")
            print(f"  - 輸入特徵數 (Pilots): {self.M}")
            print(f"  - 輸出標籤數 (Data): {self.S}")

    def extract_training_samples(self, h_block_ls):
        """
        核心創新點：從 Block Pilot (全頻帶估計) 中提取訓練樣本
        利用滑動窗口 (Sliding Window) 增加樣本量
        """
        K = len(h_block_ls)
        X_train = [] # Inputs (Pilots)
        Y_train = [] # Labels (Data)
        
        # 滑動窗口遍歷整個頻帶
        # 只要窗口不超出邊界，每一個位置都可以作為一個訓練樣本
        # 這對應論文提及的 "K - (M-1)Df + 1" 個樣本對
        num_samples = K - self.group_len + 1
        
        for i in range(num_samples):
            # 提取當前窗口的信道片段
            window = h_block_ls[i : i + self.group_len]
            
            # 提取特徵 (偽導頻) 和 標籤 (偽數據)
            x_sample = window[self.pilot_rel_idx]
            y_sample = window[self.data_rel_idx]
            
            X_train.append(x_sample)
            Y_train.append(y_sample)
            
        return np.array(X_train).T, np.array(Y_train).T

    def train(self, h_block_ls):
        """
        線上訓練步驟
        :param h_block_ls: 接收到的 Block Pilot 的 LS 估計 (長度 K)
        """
        # 1. 構建訓練數據集
        X, Y = self.extract_training_samples(h_block_ls)
        
        # 2. 計算權重矩陣 Wd
        # 公式: W = Y * pinv(X)
        # 形狀: (S, M) = (S, N_samp) * (N_samp, M)
        self.W_d = Y @ np.linalg.pinv(X)
        
        # 計算訓練誤差 (NMSE) 用於監控
        Y_pred = self.W_d @ X
        mse = np.mean(np.abs(Y_pred - Y)**2)

        if self.test:
            print(f"線上訓練完成。樣本數: {X.shape[1]}, 訓練 MSE: {mse:.6f}")

    def estimate(self, h_pilot_vec, K_total):
        """
        修正版：加入循環邊界處理 (Wrap-around)
        :param h_pilot_vec: 導頻位置的 LS 估計
        :param K_total: 系統總子載波數 (e.g., 64)
        :return: 完整的信道估計 (長度 K_total)
        """
        if self.W_d is None:
            raise ValueError("Model not trained yet!")
            
        n_pilots = len(h_pilot_vec)
        
        # 1. 初始化完整的容器 (長度為 64)
        h_est_full = np.zeros(K_total, dtype=complex)
        
        # 2. 填入導頻位置 (假設導頻從 0 開始，間隔固定)
        # pilot_indices: [0, 4, 8, ..., 60]
        pilot_freq_indices = np.arange(0, K_total, self.Df)
        h_est_full[pilot_freq_indices] = h_pilot_vec
        
        # 3. 循環遍歷每一對導頻 (包含最後一個導頻到第一個導頻的區間)
        for i in range(n_pilots):
            # 取出當前導頻 和 下一個導頻 (使用 % 運算實現循環)
            # 當 i 為最後一個時，(i+1) 會回到 0
            p_curr = h_pilot_vec[i]
            p_next = h_pilot_vec[(i + 1) % n_pilots]
            
            # 構建輸入特徵 (針對 M=2 的情況: [左導頻, 右導頻])
            x_input = np.array([p_curr, p_next])
            
            # 使用訓練好的 LML 模型預測中間的數據
            y_pred = self.W_d @ x_input
            
            # 計算要填入的絕對頻率索引
            start_freq_idx = pilot_freq_indices[i]
            
            # data_rel_idx 是 [1, 2, 3] (當 Df=4)
            # 使用 % K_total 確保 61+1, 61+2... 會正確映射 (雖然這裡是線性填入)
            # 對於最後一段：60 + [1,2,3] -> [61, 62, 63]，不會溢位，但習慣加上 %
            data_abs_indices = (start_freq_idx + self.data_rel_idx) % K_total
            
            h_est_full[data_abs_indices] = y_pred
            
        return h_est_full

# --- 整合測試 (Main Loop) ---
if __name__ == "__main__":
    # 參數設置
    K = 64
    CP = 16
    Df = 4
    M = 2 # 每個 Group 2 個導頻 (線性插值結構)
    
    # 1. 系統與模型初始化
    sys = OFDMSystem(K=K, CP=CP, pilot_spacing=Df, channel_taps=8)
    lml = LML_Estimator(M=M, pilot_spacing=Df, test=True)
    
    # 生成真實信道 (假設在整個過程中變化不大，或為 Block Fading)
    h_t, h_f_true = sys.generate_random_channel()
    
    # ==========================
    # Phase 1: Online Training
    # ==========================
    print("\n--- Phase 1: Online Training (Block Pilot) ---")
    # 發送 Block Pilot (全導頻)
    tx_train, x_train_f = sys.transmitter(payload_bits=[], mode='training')
    rx_train = sys.channel_propagation(tx_train, h_t, snr_db=20) # 高 SNR 進行訓練
    y_train_f = sys.receiver_processing(rx_train)
    
    # 計算 LS 估計 (作為 Ground Truth 的有噪版本)
    h_block_ls = y_train_f / x_train_f
    
    # 訓練 LML 模型
    lml.train(h_block_ls)
    
    # ==========================
    # Phase 2: Data Transmission
    # ==========================
    print("\n--- Phase 2: Data Transmission (Comb Pilot) ---")
    # 發送 Data Symbol (梳狀導頻)
    bits = np.random.randint(0, 2, sys.n_data * 2)
    tx_data, x_data_f = sys.transmitter(bits, mode='data')
    rx_data = sys.channel_propagation(tx_data, h_t, snr_db=15) # 測試 SNR
    y_data_f = sys.receiver_processing(rx_data)
    
    # 2.1 提取導頻位置的 LS 估計
    # 接收到的信號在導頻位置的值 / 發送的導頻值
    # 注意：我們需要知道發送的導頻值。在 transmitter 中我們設為 1+0j
    # 實際系統中 Rx 知道 Pilot Sequence
    pilot_rx = y_data_f[sys.pilot_indices]
    pilot_tx = x_data_f[sys.pilot_indices]
    h_pilot_ls = pilot_rx / pilot_tx
    
    # 2.2 LML 估計
    h_est_lml = lml.estimate(h_pilot_ls, K)
    
    # 2.3 對比方案：傳統線性插值 (Linear Interpolation)
    f_interp = interp1d(sys.pilot_indices, h_pilot_ls, kind='linear', fill_value="extrapolate")
    h_est_linear = f_interp(np.arange(K))
    
    # ==========================
    # Phase 3: Performance Check
    # ==========================
    # 計算 NMSE (Normalized Mean Square Error)
    nmse_lml = 10 * np.log10(np.mean(np.abs(h_f_true - h_est_lml)**2) / np.mean(np.abs(h_f_true)**2))
    nmse_linear = 10 * np.log10(np.mean(np.abs(h_f_true - h_est_linear)**2) / np.mean(np.abs(h_f_true)**2))
    
    print(f"\n性能對比 (SNR=15dB):")
    print(f"  - Linear Interpolation NMSE: {nmse_linear:.2f} dB")
    print(f"  - LML Estimator NMSE:        {nmse_lml:.2f} dB")
    
    # 繪圖
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(np.abs(h_f_true), 'k-', label='True Channel', linewidth=2, alpha=0.6)
    plt.plot(sys.pilot_indices, np.abs(h_pilot_ls), 'ro', label='Pilots (LS)')
    plt.plot(np.abs(h_est_linear), 'g--', label='Linear Interp')
    plt.plot(np.abs(h_est_lml), 'b-.', label='LML Est')
    plt.title('Magnitude Response')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    # 觀察局部細節
    plt.plot(np.abs(h_f_true)[10:30], 'k-', label='True', marker='.')
    plt.plot(np.abs(h_est_linear)[10:30], 'g--', label='Linear', marker='x')
    plt.plot(np.abs(h_est_lml)[10:30], 'b-.', label='LML', marker='+')
    plt.title('Zoom-in Detail (Subcarriers 10-30)')
    plt.grid(True)
    plt.legend()
    
    plt.show()