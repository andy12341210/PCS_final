import numpy as np
import matplotlib.pyplot as plt

class OFDMSystem:
    def __init__(self, K=64, CP=16, pilot_spacing=4, channel_taps=5):
        """
        初始化 OFDM 系統
        :param K: 子載波總數 (FFT size)
        :param CP: 循環前綴長度
        :param pilot_spacing: 導頻間隔 (Df)
        :param channel_taps: 多徑信道的路徑數 (用於生成隨機信道)
        """
        self.K = K
        self.CP = CP
        self.P_spacing = pilot_spacing
        self.L = channel_taps
        
        # --- 定義導頻位置 (Comb-type) [參考論文 Fig. 1] ---
        # 假設導頻從索引 0 開始: 0, 4, 8, ...
        self.pilot_indices = np.arange(0, K, pilot_spacing)
        
        # 定義數據位置 (排除導頻位置)
        self.all_indices = np.arange(K)
        self.data_indices = np.setdiff1d(self.all_indices, self.pilot_indices)
        
        # 計算導頻和數據的數量
        self.n_pilots = len(self.pilot_indices)
        self.n_data = len(self.data_indices)

        # print(f"系統初始化: K={K}, CP={CP}, Pilot Spacing={pilot_spacing}")
        # print(f"導頻子載波數量: {self.n_pilots}, 數據子載波數量: {self.n_data}")

    def qpsk_mod(self, bits):
        """簡單的 QPSK 調變 (映射: 00->1+j, 01->-1+j, ... normalized)"""
        # 將位元流 reshape 成 2 bits 一組
        symbols = []
        for i in range(0, len(bits), 2):
            b0, b1 = bits[i], bits[i+1]
            # 映射邏輯 (可根據需求調整)
            real = 1/np.sqrt(2) * (1 - 2*b0)
            imag = 1/np.sqrt(2) * (1 - 2*b1)
            symbols.append(real + 1j*imag)
        return np.array(symbols)

    def generate_random_channel(self):
        """
        生成隨機多徑衰落信道 (Time Domain)
        對應論文公式 (1): h^t
        """
        # 生成 L 徑的瑞利衰落信道 (複高斯隨機變量)
        h_time = (np.random.randn(self.L) + 1j * np.random.randn(self.L)) / np.sqrt(2)
        # 歸一化能量
        h_time = h_time / np.linalg.norm(h_time)
        
        # 計算頻域響應 H^f (用於驗證和計算理想 LS)
        # padding 到 K 長度進行 FFT
        h_freq = np.fft.fft(h_time, n=self.K)
        return h_time, h_freq

    def transmitter(self, payload_bits, mode='data'):
        """
        OFDM 發射機
        :param payload_bits: 輸入的位元流
        :param mode: 'data' (梳狀導頻) 或 'training' (全導頻/Block Pilot)
        :return: tx_signal_time (時域信號), x_freq (頻域符號-用於Debug)
        """
        # 1. 準備頻域符號容器
        x_freq = np.zeros(self.K, dtype=complex)
        
        # 2. 生成導頻符號 (通常是已知的隨機序列或固定值)
        # 這裡簡化為全 1+0j，實際應隨機生成但接收端已知
        pilot_symbols = np.ones(self.K, dtype=complex) # 簡化
        
        if mode == 'training':
            # --- Block Pilot 模式 (論文用於線上訓練) ---
            # 所有子載波都放導頻 
            x_freq = pilot_symbols 
            
        else: # mode == 'data'
            # --- 梳狀導頻模式 (論文 Fig. 1) ---
            # 1. 映射數據位元到 QPSK 符號
            if len(payload_bits) > 0:
                data_symbols = self.qpsk_mod(payload_bits)
                # 確保數據長度匹配
                if len(data_symbols) != self.n_data:
                    raise ValueError(f"數據位元不足或過多，需要 {self.n_data} 個符號")
                
                # 填入數據
                x_freq[self.data_indices] = data_symbols
            
            # 2. 填入導頻
            # 注意：論文提到導頻位置是梳狀分布
            x_freq[self.pilot_indices] = pilot_symbols[self.pilot_indices]
            
        # 3. IFFT (頻域 -> 時域)
        x_time = np.fft.ifft(x_freq) * np.sqrt(self.K) # 能量歸一化因子視習慣而定
        
        # 4. 加入 CP (Cyclic Prefix)
        x_time_cp = np.concatenate([x_time[-self.CP:], x_time])
        
        return x_time_cp, x_freq

    def channel_propagation(self, tx_signal, h_time, snr_db):
        """
        通過信道並加入噪聲
        對應論文公式 (1): y^t = x^t * h^t + z
        """
        # 1. 線性卷積 (Linear Convolution)
        # 注意：實際物理信道是線性卷積，去除CP後等效為循環卷積
        rx_signal_clean = np.convolve(tx_signal, h_time, mode='full')
        
        # 截斷到發送長度 (模擬接收窗口)
        rx_signal_clean = rx_signal_clean[:len(tx_signal)]
        
        # 2. 計算信號功率並加入 AWGN 噪聲
        sig_power = np.mean(np.abs(rx_signal_clean)**2)
        snr_linear = 10**(snr_db / 10)
        noise_power = sig_power / snr_linear
        noise = (np.random.randn(len(rx_signal_clean)) + 1j*np.random.randn(len(rx_signal_clean))) / np.sqrt(2)
        noise = noise * np.sqrt(noise_power)
        
        y_time = rx_signal_clean + noise
        return y_time

    def receiver_processing(self, rx_signal):
        """
        接收機前端處理
        :return: y_freq (接收到的頻域信號), y_pilots (僅導頻位置), y_data (僅數據位置)
        """
        # 1. 移除 CP [cite: 1785]
        # 假設完美同步
        rx_signal_no_cp = rx_signal[self.CP : self.CP + self.K]
        
        # 2. FFT (時域 -> 頻域)
        y_freq = np.fft.fft(rx_signal_no_cp) / np.sqrt(self.K)
        
        # 對應論文公式 (2): y^f = X^f h^f + z
        return y_freq

# --- 測試代碼 ---
if __name__ == "__main__":
    # 1. 實例化系統
    ofdm = OFDMSystem(K=64, CP=16, pilot_spacing=4)
    
    # 2. 生成一個隨機信道
    h_time, h_freq_true = ofdm.generate_random_channel()
    
    # 3. 模擬傳輸一個 "Block Pilot" (訓練用)
    tx_train, x_train_freq = ofdm.transmitter(payload_bits=[], mode='training')
    rx_train = ofdm.channel_propagation(tx_train, h_time, snr_db=20)
    y_train_freq = ofdm.receiver_processing(rx_train)
    
    # 簡單驗證 LS 估計 (Block Pilot 所有位置都有信號)
    h_ls_est = y_train_freq / x_train_freq
    
    # 4. 模擬傳輸一個 "Data Symbol" (測試用)
    # 生成隨機 bits
    n_bits = ofdm.n_data * 2 # QPSK = 2 bits per symbol
    bits = np.random.randint(0, 2, n_bits)
    tx_data, x_data_freq = ofdm.transmitter(bits, mode='data')
    rx_data = ofdm.channel_propagation(tx_data, h_time, snr_db=20)
    y_data_freq = ofdm.receiver_processing(rx_data)
    
    # 繪圖驗證 (幅度響應)
    plt.figure(figsize=(10, 4))
    plt.plot(np.abs(h_freq_true), label='True Channel')
    plt.plot(np.abs(h_ls_est), '--', label='LS Estimate (from Block Pilot)')
    plt.title('Channel Frequency Response (Magnitude)')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print("OFDM Base System check passed.")