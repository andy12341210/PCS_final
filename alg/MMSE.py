import numpy as np

# ==========================================
# 2. MMSE (統計估測器)
# ==========================================
class MMSE:
    def __init__(self, system):
        self.sys = system
        self.W_mmse = None

    def calculate_weights(self, snr_lin):
        """計算 LMMSE 權重矩陣 W"""
        pilot_pos = np.array([0, 3]) # Local group relative positions
        data_pos = np.array([1, 2])

        # Correlation function R(delta_k)
        def correlation(delta_k):
            return np.sum(self.sys.path_powers * np.exp(-1j * 2 * np.pi * delta_k * self.sys.path_delays * self.sys.Fs / self.sys.K_total))

        # R_hp_hp (2x2)
        R_hp_hp = np.zeros((self.sys.M, self.sys.M), dtype=complex)
        for i in range(self.sys.M):
            for j in range(self.sys.M):
                R_hp_hp[i, j] = correlation(pilot_pos[i] - pilot_pos[j])

        # R_hd_hp (2x2)
        R_hd_hp = np.zeros((self.sys.S, self.sys.M), dtype=complex)
        for i in range(self.sys.S):
            for j in range(self.sys.M):
                R_hd_hp[i, j] = correlation(data_pos[i] - pilot_pos[j])

        sigma_n2 = 1.0 / snr_lin
        inv_term = np.linalg.inv(R_hp_hp + sigma_n2 * np.eye(self.sys.M))
        self.W_mmse = np.dot(R_hd_hp, inv_term)

    def estimate(self, h_pilots_noisy):
        """使用預計算的權重進行估測"""
        h_est_data = np.zeros(len(self.sys.data_indices), dtype=complex)
        
        for g in range(self.sys.num_groups):
            p_vec = h_pilots_noisy[g : g+2] # 取相鄰兩個 pilot
            d_hat = np.dot(self.W_mmse, p_vec)
            
            start_idx = g * self.sys.S
            h_est_data[start_idx : start_idx + self.sys.S] = d_hat
            
        return h_est_data