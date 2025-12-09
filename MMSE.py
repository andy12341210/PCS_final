import numpy as np
import os

class MMSE_Estimator:
    def __init__(self, ofdm_system, n_samples=10000, test = True):
        """
        初始化 MMSE 估計器 (帶有自動存檔功能)
        :param ofdm_system: OFDMSystem 實例
        :param n_samples: 用於統計的樣本數 (僅在沒有存檔時使用)
        """
        self.sys = ofdm_system
        self.K = ofdm_system.K
        self.pilot_indices = ofdm_system.pilot_indices
        self.n_pilots = len(self.pilot_indices)
        
        # 產生一個唯一的檔案名稱，包含關鍵參數
        # 這樣當您修改 K 或 多徑數(L) 時，不會讀錯舊檔案
        self.cache_filename = f"mmse_stats_K{self.K}_CP{self.sys.CP}_L{self.sys.L}_Samples{n_samples}.npz"
        
        self.R_hh = None
        self.R_h_hp = None
        self.R_hp_hp = None
        
        # 檢查是否有存檔
        if os.path.exists(self.cache_filename):
            if test:
                print(f"MMSE Estimator: 發現存檔 '{self.cache_filename}'，正在讀取...")
            self.load_statistics()
        else:
            if test:
                print(f"MMSE Estimator: 未發現存檔，開始計算統計矩陣 (n={n_samples})...")
            self.compute_channel_statistics(n_samples)
            self.save_statistics()
            if test:
                print("MMSE Estimator: 計算完成並已存檔。")

    def compute_channel_statistics(self, n_samples):
        """
        蒙地卡羅統計 (耗時操作)
        """
        H_samples = []
        for _ in range(n_samples):
            _, h_freq = self.sys.generate_random_channel()
            H_samples.append(h_freq)
            
        H_samples = np.array(H_samples).T
        
        # 計算 R_hh
        self.R_hh = (H_samples @ H_samples.conj().T) / n_samples
        
        # 提取子矩陣
        self.R_hp_hp = self.R_hh[np.ix_(self.pilot_indices, self.pilot_indices)]
        self.R_h_hp = self.R_hh[:, self.pilot_indices]

    def save_statistics(self):
        """將矩陣存入 .npz 檔案"""
        np.savez(self.cache_filename, 
                 R_hh=self.R_hh, 
                 R_h_hp=self.R_h_hp, 
                 R_hp_hp=self.R_hp_hp)

    def load_statistics(self):
        """從 .npz 檔案讀取矩陣"""
        data = np.load(self.cache_filename)
        self.R_hh = data['R_hh']
        self.R_h_hp = data['R_h_hp']
        self.R_hp_hp = data['R_hp_hp']

    def estimate(self, h_pilot_ls, snr_db):
        """
        執行 MMSE 信道估計
        """
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = 1.0 / snr_linear
        
        # Regularization with Noise Power
        regularization = noise_power * np.eye(self.n_pilots)
        
        # Wiener Filter
        # W = R_h_hp * inv(R_hp_hp + sigma^2 * I)
        term_inv = np.linalg.inv(self.R_hp_hp + regularization)
        W = self.R_h_hp @ term_inv
        
        h_est_mmse = W @ h_pilot_ls
        return h_est_mmse
    
# ==========================================
# 单元測試代碼 (Unit Testing)
# ==========================================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.interpolate import interp1d
    
    # 確保引用路徑正確，如果都在同一目錄下則沒問題
    try:
        from OFDMsystem import OFDMSystem
    except ImportError:
        print("錯誤：找不到 OFDMsystem.py，請確保檔案在同一目錄下。")
        exit()

    def test_mmse_single_shot():
        print("\n--- 開始 MMSE 單元測試 ---")
        
        # 1. 設置參數
        K = 64
        CP = 16
        Df = 4 # Pilot Spacing
        test_snr = 15 # 測試用的 SNR (dB)
        
        # 2. 初始化系統
        # 故意增加 channel_taps (如 8 或 10) 讓信道頻率選擇性更強，這樣 MMSE 的效果才明顯
        sys = OFDMSystem(K=K, CP=CP, pilot_spacing=Df, channel_taps=10)
        
        print("初始化 MMSE Estimator (這可能會觸發統計計算)...")
        mmse_est = MMSE_Estimator(sys, n_samples=5000) # 測試用 5000 次即可
        
        # 3. 生成一次隨機信道與傳輸
        h_time, h_true = sys.generate_random_channel()
        
        # 模擬發送導頻 (為了方便，這裡發送 Block Pilot 並提取導頻位置)
        tx_sig, tx_freq = sys.transmitter([], mode='training')
        rx_sig = sys.channel_propagation(tx_sig, h_time, snr_db=test_snr)
        y_freq = sys.receiver_processing(rx_sig)
        
        # 獲取導頻位置的 LS 估計 (含噪聲)
        h_ls_full = y_freq / tx_freq
        h_ls_pilots = h_ls_full[sys.pilot_indices]
        
        # 4. 運行 MMSE 估計
        h_est_mmse = mmse_est.estimate(h_ls_pilots, snr_db=test_snr)
        
        # 5. 運行對照組：線性插值 (Linear Interpolation)
        f_interp = interp1d(sys.pilot_indices, h_ls_pilots, kind='linear', fill_value="extrapolate")
        h_est_linear = f_interp(np.arange(K))
        
        # 6. 計算誤差 (MSE)
        mse_mmse = np.mean(np.abs(h_true - h_est_mmse)**2)
        mse_linear = np.mean(np.abs(h_true - h_est_linear)**2)
        
        print(f"\n測試結果 (SNR={test_snr}dB):")
        print(f"  - Linear Interp MSE: {mse_linear:.6f}")
        print(f"  - MMSE Estimator MSE: {mse_mmse:.6f}")
        
        if mse_mmse < mse_linear:
            print("  => 通過！MMSE 表現優於線性插值。")
        else:
            print("  => 警告！MMSE 表現不如預期 (可能是 SNR 太低或信道太簡單)。")

        # 7. 繪圖驗證
        plt.figure(figsize=(10, 6))
        
        # 幅度響應
        plt.subplot(2, 1, 1)
        plt.plot(np.abs(h_true), 'k-', label='True Channel', linewidth=2, alpha=0.6)
        plt.plot(sys.pilot_indices, np.abs(h_ls_pilots), 'ro', label='Noisy Pilots')
        plt.plot(np.abs(h_est_linear), 'b--', label='Linear Interp')
        plt.plot(np.abs(h_est_mmse), 'g-', label='MMSE Estimate')
        plt.title(f'Channel Magnitude Response (SNR={test_snr}dB)')
        plt.legend()
        plt.grid(True)
        
        # 相位響應 (選看)
        plt.subplot(2, 1, 2)
        plt.plot(np.angle(h_true), 'k-', label='True', alpha=0.6)
        plt.plot(np.angle(h_est_mmse), 'g-', label='MMSE')
        plt.title('Channel Phase Response')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    # 執行測試
    test_mmse_single_shot()