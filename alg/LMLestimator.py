import numpy as np

class LMLestimator:
    def __init__(self, system):
        self.sys = system
        self.W_hat = None
        self.X_train_stored = None
        self.Y_train_stored = None

    def _generate_window_data(self, h_ls_block):
        X_list = []
        Y_list = []
        window_size = (self.sys.M - 1) * self.sys.Df + 1 
        num_samples = self.sys.K_active - window_size + 1

        for i in range(num_samples):
            vec = h_ls_block[i : i + window_size]
            # 簡單過濾 Inf
            if np.any(np.isinf(vec)) or np.any(np.isnan(vec)):
                continue

            x_sample = vec[[0, -1]]
            y_sample = vec[1:-1]
            X_list.append(x_sample)
            Y_list.append(y_sample)
        
        if len(X_list) == 0:
            return np.zeros((self.sys.M, 0)), np.zeros((self.sys.S, 0))

        return np.array(X_list).T, np.array(Y_list).T

    def _solve_weights(self, X, Y):
        """
        [修改] 使用標準 Inverse 而非 Pinv，以誠實反映 Error Propagation。
        Formula: W = Y * X^H * (X * X^H)^-1
        """
        if X.shape[1] == 0:
            return self.W_hat if self.W_hat is not None else np.zeros((self.sys.S, self.sys.M), dtype=complex)
            
        try:
            # 1. 計算自相關矩陣 R_xx = X * X^H
            # 加入極小的 regularization 防止矩陣奇異 (Singular)，但不要太大以免掩蓋錯誤
            reg = 1e-8 * np.eye(X.shape[0]) 
            R_xx = np.dot(X, X.conj().T) + reg
            
            # 2. 計算互相關矩陣 R_yx = Y * X^H
            R_yx = np.dot(Y, X.conj().T)
            
            # 3. 使用標準 inv 求解 (這在數據很髒時會比 pinv 更不穩定，符合預期)
            R_xx_inv = np.linalg.inv(R_xx)
            W = np.dot(R_yx, R_xx_inv)
            return W

        except np.linalg.LinAlgError:
            # 如果矩陣真的壞掉，回傳零矩陣
            return np.zeros((Y.shape[0], X.shape[0]), dtype=complex)

    def train(self, h_ls_block):
        self.X_train_stored, self.Y_train_stored = self._generate_window_data(h_ls_block)
        self.W_hat = self._solve_weights(self.X_train_stored, self.Y_train_stored)
    
    def train_with_data(self, X, Y):
        self.W_hat = self._solve_weights(X, Y)

    def update_ddtdg(self, h_ls_payload, forgetting_factor=0.0):
        X_new, Y_new = self._generate_window_data(h_ls_payload)
        
        if self.X_train_stored is not None:
            X_old = self.X_train_stored * forgetting_factor
            Y_old = self.Y_train_stored * forgetting_factor
        else:
            X_old, Y_old = None, None

        if X_old is not None and X_new.shape[1] > 0:
            X_combined = np.hstack([X_old, X_new])
            Y_combined = np.hstack([Y_old, Y_new])
        elif X_new.shape[1] > 0:
            X_combined = X_new
            Y_combined = Y_new
        else:
            X_combined = self.X_train_stored
            Y_combined = self.Y_train_stored
        
        return self._solve_weights(X_combined, Y_combined)

    def estimate(self, h_pilots_noisy):
        return self.estimate_with_W(h_pilots_noisy, self.W_hat)

    def estimate_with_W(self, h_pilots_noisy, W):
        if W is None:
            return np.zeros(len(self.sys.data_indices), dtype=complex)

        h_est_data = np.zeros(len(self.sys.data_indices), dtype=complex)
        
        for g in range(self.sys.num_groups):
            p_vec = h_pilots_noisy[g : g+2]
            d_hat = np.dot(W, p_vec)
            start_idx = g * self.sys.S
            h_est_data[start_idx : start_idx + self.sys.S] = d_hat
            
        return h_est_data