import numpy as np
from scipy.interpolate import interp1d

class OFDMSystem:
    def __init__(self):
        # 系統參數
        self.K_total = 512
        self.K_active = 410
        self.CP = 128
        self.Df = 3
        self.M = 2
        self.S = self.Df - 1 # 2
        self.Fs = 20e6

        # Channel Model: Pedestrian B
        self.path_delays = np.array([0, 200, 800, 1200, 2300, 3700]) * 1e-9
        self.path_powers_db = np.array([0, -0.9, -4.9, -8.0, -7.8, -23.9])
        self.path_powers = 10 ** (self.path_powers_db / 10)
        self.path_powers /= np.sum(self.path_powers)

        # Indices
        self.pilot_indices = np.arange(0, self.K_active, self.Df)
        self.num_groups = len(self.pilot_indices) - 1
        self.data_indices = np.setdiff1d(np.arange(self.pilot_indices[-1]+1), self.pilot_indices)

        self.valid_range = self.pilot_indices[-1] + 1

    def generate_channel(self):
        n_taps = len(self.path_delays)
        h_t = np.zeros(self.CP, dtype=complex)
        path_gains = (np.random.randn(n_taps) + 1j * np.random.randn(n_taps)) / np.sqrt(2)
        path_gains = path_gains * np.sqrt(self.path_powers)
        sample_indices = np.round(self.path_delays * self.Fs).astype(int)
        
        for i, idx in enumerate(sample_indices):
            if idx < len(h_t):
                h_t[idx] += path_gains[i]

        H_f_full = np.fft.fft(h_t, n=self.K_total)
        return H_f_full[0:self.K_active]

    def qpsk_modulation(self, num_symbols):
        """回傳 (Symbols, Integers)"""
        ints = np.random.randint(0, 4, num_symbols)
        s = np.zeros(num_symbols, dtype=complex)
        s[ints==0] = 1+1j
        s[ints==1] = -1+1j
        s[ints==2] = -1-1j
        s[ints==3] = 1-1j
        return s / np.sqrt(2), ints

    def add_noise(self, signal, snr_db):
        sig_power = np.mean(np.abs(signal)**2)
        noise_power = sig_power / (10**(snr_db/10))
        noise = (np.random.randn(*signal.shape) + 1j*np.random.randn(*signal.shape)) / np.sqrt(2)
        return signal + np.sqrt(noise_power) * noise, noise_power

    def calc_nmse(self, h_true, h_est):
        return np.mean(np.abs(h_true - h_est)**2) / np.mean(np.abs(h_true)**2)

    # --- 新增: 非線性失真 (Clipping) ---
    def nonlinear_distortion(self, signal):
        amplitude = np.abs(signal)
        rms = np.sqrt(np.mean(amplitude**2))
        A = rms  # Threshold = RMS
        mask = amplitude > A
        distorted_signal = signal.copy()
        # Clipping: A * e^(j*phi)
        distorted_signal[mask] = A * (signal[mask] / amplitude[mask])
        return distorted_signal

    # --- 新增: 解調與 BER 計算 ---
    def demodulate_qpsk(self, received_signal):
        decoded_ints = np.zeros(len(received_signal), dtype=int)
        re = received_signal.real
        im = received_signal.imag
        decoded_ints[(re >= 0) & (im >= 0)] = 0
        decoded_ints[(re < 0) & (im >= 0)] = 1
        decoded_ints[(re < 0) & (im < 0)] = 2
        decoded_ints[(re >= 0) & (im < 0)] = 3
        return decoded_ints

    def calculate_ber(self, sent_ints, dec_ints):
        # 簡單計算: XOR 差異 bit 數 / 總 bit 數 (QPSK 2 bits/symbol)
        # Mapping: 0(00), 1(01), 2(11), 3(10) -> 這裡簡化假設 0,1,2,3 對應二進制
        # 嚴謹應使用 Gray Code，此處依題目邏輯實作
        diff = sent_ints ^ dec_ints
        bit_errors = 0
        for d in diff:
            bit_errors += bin(d).count('1')
        return bit_errors / (len(sent_ints) * 2)

    def ls_interpolation(self, h_pilot_noisy):
        """Baseline: LS + Linear Interpolation"""
        # 分開實部虛部插值
        f_real = interp1d(self.pilot_indices, h_pilot_noisy.real, kind='linear', fill_value="extrapolate")
        f_imag = interp1d(self.pilot_indices, h_pilot_noisy.imag, kind='linear', fill_value="extrapolate")
        
        # 對整個有效範圍進行插值
        h_est_full = f_real(np.arange(self.valid_range)) + 1j * f_imag(np.arange(self.valid_range))
        
        # 只回傳 Data 位置的估測值以計算 NMSE
        return h_est_full[self.data_indices]

    def get_training_dataset(self, snr_db, dataset_size_limit=None):
        H_true = self.generate_channel()
        x_block, _ = self.qpsk_modulation(self.K_active) # [修正] 接收兩個回傳值
        y_block_rx_clean = H_true * x_block
        y_block_rx, _ = self.add_noise(y_block_rx_clean, snr_db)
        
        h_ls_block = y_block_rx / x_block

        X_list, Y_noisy_list, Y_clean_list = [], [], []
        window_size = (self.M - 1) * self.Df + 1 
        num_samples = self.K_active - window_size + 1
        
        for i in range(num_samples):
            vec_ls = h_ls_block[i : i + window_size]
            vec_true = H_true[i : i + window_size]
            
            x_sample = vec_ls[[0, -1]]
            y_sample_noisy = vec_ls[1:-1]
            y_sample_clean = vec_true[1:-1]

            X_list.append(x_sample)
            Y_noisy_list.append(y_sample_noisy)
            Y_clean_list.append(y_sample_clean)

            if dataset_size_limit is not None and len(X_list) >= dataset_size_limit:
                break
        
        return np.array(X_list).T, np.array(Y_noisy_list).T, np.array(Y_clean_list).T