import numpy as np

# ==========================================
# 3. LMLestimator (Proposed Method)
# ==========================================
class LMLestimator:
    def __init__(self, system):
        self.sys = system
        self.W_hat = None

    def train(self, h_ls_block):
        """
        PATDG Training: 使用 Block Pilot 的 LS 估測結果學習 W
        h_ls_block: 全導頻符號的 LS 通道響應
        """
        X_train = []
        Y_train = []

        # Sliding window size = (M-1)Df + 1 = 4
        window_size = (self.sys.M - 1) * self.sys.Df + 1 
        num_samples = self.sys.K_active - window_size + 1

        for i in range(num_samples):
            vec = h_ls_block[i : i + window_size]
            x_sample = vec[[0, -1]] # Pseudo-Pilots (頭尾)
            y_sample = vec[1:-1]    # Pseudo-Data (中間)
            
            X_train.append(x_sample)
            Y_train.append(y_sample)

        X_train = np.array(X_train).T # (2, T)
        Y_train = np.array(Y_train).T # (2, T)

        # W = Y * X_pinv
        X_pinv = np.linalg.pinv(X_train)
        self.W_hat = np.dot(Y_train, X_pinv)
    
    def train_with_data(self, X, Y):
        """新增方法: 直接使用給定的 X, Y 計算權重"""
        try:
            X_pinv = np.linalg.pinv(X)
            self.W_hat = np.dot(Y, X_pinv)
        except np.linalg.LinAlgError:
            self.W_hat = np.zeros((Y.shape[0], X.shape[0]), dtype=complex)

    def estimate(self, h_pilots_noisy):
        """使用學習到的 W_hat 進行估測"""
        h_est_data = np.zeros(len(self.sys.data_indices), dtype=complex)
        
        for g in range(self.sys.num_groups):
            p_vec = h_pilots_noisy[g : g+2]
            d_hat = np.dot(self.W_hat, p_vec)
            
            start_idx = g * self.sys.S
            h_est_data[start_idx : start_idx + self.sys.S] = d_hat
            
        return h_est_data