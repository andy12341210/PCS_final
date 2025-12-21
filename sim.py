import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 引用您的自定義模組
from OFDMsystem import OFDMSystem
from alg.LMLestimator import LMLestimator
from alg.MMSE import MMSE
from alg.Neural_estimators import ChannelNet, ELM_Estimator, complex_to_real_torch, real_torch_to_complex

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_simulation_fig5():
    # 初始化物件
    ofdm = OFDMSystem()
    mmse_est = MMSE(ofdm)
    lml_est = LMLestimator(ofdm)

    snr_db_range = np.arange(-10, 20, 4)
    num_monte_carlo = 1000

    # 準備儲存結果的字典
    nmse_results = {'ls': [], 'mmse': [], 'proposed': [], 'ddtdg': []}
    ber_results = {'ls': [], 'mmse': [], 'proposed': [], 'ddtdg': []}

    print(f"Starting Figure 5 Simulation...")

    for snr_db in snr_db_range:
        snr_lin = 10**(snr_db/10)
        
        # 累積誤差變數
        loss_nmse = {'ls': 0, 'mmse': 0, 'proposed': 0, 'ddtdg': 0}
        loss_ber = {'ls': 0, 'mmse': 0, 'proposed': 0, 'ddtdg': 0}
        
        # 記錄錯誤數 (Debug)
        debug_errors = 0

        mmse_est.calculate_weights(snr_lin)

        for mc in range(num_monte_carlo):
            H_true = ofdm.generate_channel()

            # ==========================================
            # Phase 1: Training (PATDG - Only Pilots)
            # ==========================================
            x_train, _ = ofdm.qpsk_modulation(ofdm.K_active)
            y_train_clean = H_true * x_train
            y_train_rx, _ = ofdm.add_noise(y_train_clean, snr_db)
            h_ls_train = y_train_rx / x_train
            
            lml_est.train(h_ls_train)

            # ==========================================
            # Phase 2: Testing Phase
            # ==========================================
            x_payload, x_ints_true = ofdm.qpsk_modulation(ofdm.K_active)
            y_payload_clean = H_true * x_payload
            
            # [重要] 確保雜訊是真的加上去了
            y_payload_rx, _ = ofdm.add_noise(y_payload_clean, snr_db)

            # 導頻 LS 估測 (Input for Estimators)
            noise_power_est = 1.0 / snr_lin # Theoretical Noise Power for MMSE
            
            # 手動添加導頻處的雜訊 (確保與 Payload 雜訊一致性)
            # 這裡我們直接從 y_payload_rx 取樣，而不是重新生成雜訊，這樣更物理真實
            # 假設 x_payload 在導頻位置就是 x_payload[pilot_indices]
            # 這樣 h_ls_pilots 就會包含真實的 channel noise
            h_ls_pilots = y_payload_rx[ofdm.pilot_indices] / x_payload[ofdm.pilot_indices]

            # Ground Truth
            h_true_data = H_true[ofdm.data_indices]
            y_data_rx = y_payload_rx[ofdm.data_indices]
            x_ints_data_true = x_ints_true[ofdm.data_indices]

            # --- 評估函式 ---
            def evaluate_method(h_est_data, name):
                # 1. NMSE
                loss_nmse[name] += ofdm.calc_nmse(h_true_data, h_est_data)
                
                # 2. Equalization & BER
                h_est_safe = h_est_data.copy()
                # 防止除以極小值 (但不要蓋過錯誤)
                h_est_safe[np.abs(h_est_safe) < 1e-10] = 1e-10
                
                x_hat = y_data_rx / h_est_safe
                dec_ints = ofdm.demodulate_qpsk(x_hat)
                
                errs = ofdm.calculate_ber(x_ints_data_true, dec_ints)
                loss_ber[name] += errs
                return dec_ints

            # [A] LS
            h_ls_data = ofdm.ls_interpolation(h_ls_pilots)
            evaluate_method(h_ls_data, 'ls')

            # [B] MMSE
            h_mmse_data = mmse_est.estimate(h_ls_pilots)
            evaluate_method(h_mmse_data, 'mmse')

            # [C] Proposed (PATDG)
            h_prop_data = lml_est.estimate(h_ls_pilots)
            dec_ints_prop = evaluate_method(h_prop_data, 'proposed')

            # Debug: 計算第一次 MC 的 Symbol Error
            if mc == 0:
                diff = np.sum(dec_ints_prop != x_ints_data_true)
                debug_errors = diff

            # ==========================================
            # [D] DDTDG (Honest Implementation)
            # ==========================================
            # 1. 重構訊號 (Reconstruct using Decisions)
            s_map = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
            x_rec = np.zeros(ofdm.K_active, dtype=complex)
            
            # 導頻位置用已知的 (Perfect)
            x_rec[ofdm.pilot_indices] = x_payload[ofdm.pilot_indices]
            
            # 資料位置用判決的 (Dirty) -> 如果 SNR 低，這裡會有很多錯
            x_rec[ofdm.data_indices] = s_map[dec_ints_prop] 

            # 2. 計算 Dirty LS (Full Band)
            # 為了避免除以 0，找出 x_rec 為 0 的地方 (理論上 QPSK 不會是 0，除非 index 沒填到)
            # 這裡補上非 Data 非 Pilot 的空位
            mask_valid = (x_rec != 0)
            
            h_ls_payload = np.zeros_like(y_payload_rx)
            h_ls_payload[mask_valid] = y_payload_rx[mask_valid] / x_rec[mask_valid]

            # 3. 更新權重 (Forgetting Factor = 0)
            # 強迫 estimator 使用這個充滿誤差的 h_ls_payload 進行訓練
            W_ddtdg = lml_est.update_ddtdg(h_ls_payload, forgetting_factor=0.0)

            # 4. 重新估測
            h_ddtdg_data = lml_est.estimate_with_W(h_ls_pilots, W_ddtdg)
            evaluate_method(h_ddtdg_data, 'ddtdg')

        # 平均
        for method in ['ls', 'mmse', 'proposed', 'ddtdg']:
            nmse_results[method].append(loss_nmse[method] / num_monte_carlo)
            ber_results[method].append(loss_ber[method] / num_monte_carlo)

        print(f"  SNR {snr_db}dB | "
              f"PATDG NMSE: {nmse_results['proposed'][-1]:.4f} | "
              f"DDTDG NMSE: {nmse_results['ddtdg'][-1]:.4f} | "
              f"Symbol Errors (MC0): {debug_errors}")

    return snr_db_range, nmse_results, ber_results

def run_simulation_fig6():
    # 初始化
    sys = OFDMSystem()
    mmse_est = MMSE(sys)
    
    # 建立兩個 Estimator
    lml_prop = LMLestimator(sys)
    lml_acc = LMLestimator(sys)

    # 模擬參數
    snr_db_list = [-10, 0, 10]
    dataset_sizes = [40, 80, 120, 160, 200, 240, 280, 320, 360, 400]
    num_monte_carlo = 1000

    res_proposed = np.zeros((len(snr_db_list), len(dataset_sizes)))
    res_accurate = np.zeros((len(snr_db_list), len(dataset_sizes)))
    res_mmse = np.zeros((len(snr_db_list), len(dataset_sizes)))

    print("Starting Figure 6 Simulation...")

    for s_idx, snr_db in enumerate(snr_db_list):
        snr_lin = 10**(snr_db/10)
        
        # MMSE 權重計算
        mmse_est.calculate_weights(snr_lin)
        
        print(f"  Processing SNR = {snr_db} dB")

        for d_idx, d_size in enumerate(dataset_sizes):
            loss_prop = 0
            loss_acc = 0
            loss_mmse = 0

            for mc in range(num_monte_carlo):
                # ==========================
                # 1. Training Phase
                # ==========================
                X, Y_noisy, Y_clean = sys.get_training_dataset(snr_db, dataset_size_limit=d_size)

                # Train Models
                lml_prop.train_with_data(X, Y_noisy)
                lml_acc.train_with_data(X, Y_clean)

                # ==========================
                # 2. Testing Phase
                # ==========================
                H_test = sys.generate_channel()
                
                noise_power = 1.0 / snr_lin
                noise = (np.random.randn(len(sys.pilot_indices)) + 1j*np.random.randn(len(sys.pilot_indices))) / np.sqrt(2)
                h_ls_pilots = H_test[sys.pilot_indices] + np.sqrt(noise_power) * noise

                # 準備 Ground Truth
                h_true_data_list = []
                for g in range(sys.num_groups):
                    base_idx = sys.pilot_indices[g]
                    h_true_data_list.extend(H_test[base_idx+1 : base_idx+3])
                h_true_data = np.array(h_true_data_list)

                # 執行估測
                h_est_prop = lml_prop.estimate(h_ls_pilots)
                h_est_acc = lml_acc.estimate(h_ls_pilots)
                h_est_mmse = mmse_est.estimate(h_ls_pilots)

                # 計算誤差
                valid_len = len(h_true_data)
                loss_prop += sys.calc_nmse(h_true_data, h_est_prop[:valid_len])
                loss_acc += sys.calc_nmse(h_true_data, h_est_acc[:valid_len])
                loss_mmse += sys.calc_nmse(h_true_data, h_est_mmse[:valid_len])

            # 平均
            res_proposed[s_idx, d_idx] = loss_prop / num_monte_carlo
            res_accurate[s_idx, d_idx] = loss_acc / num_monte_carlo
            res_mmse[s_idx, d_idx] = loss_mmse / num_monte_carlo
            
    return dataset_sizes, res_proposed, res_accurate, res_mmse, snr_db_list

def run_simulation_fig8_9():
    """
    Fig 8 & 9 Combined: 
    Scenario 2: 非完美同步 (STO/CFO) 下的 NMSE 與 BER 比較
    
    [修改] 橫軸改為 Eb/N0
    轉換公式 (QPSK): SNR(dB) = Eb/N0(dB) + 10*log10(2)
    """
    print("=== Running Figure 8 & 9 Simulation (STO/CFO Scenario) with Eb/N0 ===")
    
    # --- 模擬參數 ---
    # 橫軸改為 Eb/N0
    ebn0_db_list = np.arange(0, 16, 3) # 0 to 30 dB (Eb/N0)
    
    num_monte_carlo = 1000
    sto_val = 3       # STO samples
    cfo_val = 0.0    # Normalized CFO
    
    # --- 初始化系統 ---
    sys = OFDMSystem()
    sys.Df = 6
    sys.S = 5
    # 重算 Index
    sys.pilot_indices = np.arange(0, sys.K_active, sys.Df)
    sys.num_groups = len(sys.pilot_indices) - 1
    sys.data_indices = np.setdiff1d(np.arange(sys.pilot_indices[-1]+1), sys.pilot_indices)
    sys.valid_range = sys.pilot_indices[-1] + 1
    
    # --- 初始化估測器 ---
    mmse_est = MMSE(sys)
    lml_est = LMLestimator(sys)
    
    # --- 結果容器 ---
    results_nmse = {'ls': [], 'mmse': [], 'lml': []}
    results_ber  = {'ls': [], 'mmse': [], 'lml': []}
    
    # --- Eb/N0 Loop ---
    for ebn0_db in ebn0_db_list:
        # [關鍵修改] 將 Eb/N0 轉換為 SNR (dB)
        # QPSK: 2 bits/symbol => SNR = Eb/N0 + 3.01 dB
        snr_db = ebn0_db + 10 * np.log10(2)
        snr_lin = 10**(snr_db/10)
        
        # 累加器
        t_nmse_ls, t_nmse_mmse, t_nmse_lml = 0.0, 0.0, 0.0
        t_ber_ls, t_ber_mmse, t_ber_lml = 0.0, 0.0, 0.0
        
        # 1. 計算 MMSE 權重 (Standard MMSE, Mismatch under STO)
        mmse_est.calculate_weights(snr_lin)
        W_mmse = mmse_est.W_mmse
        
        # 2. LML Online Training (使用換算後的 snr_db 生成噪聲)
        train_size = 200
        X_train_list, Y_train_list = [], []
        
        for _ in range(train_size):
            H_tr = sys.generate_channel()
            H_tr_eff = sys.add_sto_cfo(H_tr, sto_val, cfo_val)
            
            x_tr, _ = sys.qpsk_modulation(sys.K_active)
            y_tr_clean = H_tr_eff * x_tr
            # 注意：這裡傳入的是 snr_db
            y_tr, _ = sys.add_noise(y_tr_clean, snr_db)
            
            h_ls_tr = y_tr / x_tr
            
            dx, dy = lml_est._generate_window_data(h_ls_tr)
            if dx.shape[1] > 0:
                X_train_list.append(dx)
                Y_train_list.append(dy)
        
        if len(X_train_list) > 0:
            X_train = np.hstack(X_train_list)
            Y_train = np.hstack(Y_train_list)
            lml_est.train_with_data(X_train, Y_train)
            
        # --- Monte Carlo Loop ---
        for mc in range(num_monte_carlo):
            H_true = sys.generate_channel()
            H_eff = sys.add_sto_cfo(H_true, sto_val, cfo_val)
            
            x_payload, x_bits = sys.qpsk_modulation(sys.K_active)
            y_clean = H_eff * x_payload
            y_rx, _ = sys.add_noise(y_clean, snr_db)
            
            # --- LS ---
            h_ls_pilots = y_rx[sys.pilot_indices] / x_payload[sys.pilot_indices]
            h_ls_data = sys.ls_interpolation(h_ls_pilots)
            
            # --- MMSE ---
            h_mmse_full = np.zeros(sys.K_active, dtype=complex)
            h_mmse_full[sys.pilot_indices] = h_ls_pilots
            for i in range(sys.num_groups):
                h_p_local = h_ls_pilots[i : i+2]
                if W_mmse is not None:
                    h_d_local = W_mmse @ h_p_local
                else:
                    h_d_local = np.zeros(sys.S, dtype=complex)
                p_idx_start = sys.pilot_indices[i]
                p_idx_end = sys.pilot_indices[i+1]
                h_mmse_full[p_idx_start+1 : p_idx_end] = h_d_local
            h_mmse_data = h_mmse_full[sys.data_indices]
            
            # --- LML ---
            h_lml_data = lml_est.estimate(h_ls_pilots)
            
            # --- Error Calculation ---
            h_true_data = H_eff[sys.data_indices]
            
            # NMSE
            t_nmse_ls += np.linalg.norm(h_ls_data - h_true_data)**2 / np.linalg.norm(h_true_data)**2
            t_nmse_mmse += np.linalg.norm(h_mmse_data - h_true_data)**2 / np.linalg.norm(h_true_data)**2
            t_nmse_lml += np.linalg.norm(h_lml_data - h_true_data)**2 / np.linalg.norm(h_true_data)**2
            
            # BER
            x_bits_data = x_bits[sys.data_indices]
            t_ber_ls += sys.calculate_ber(x_bits_data, sys.demodulate_qpsk(y_rx[sys.data_indices] / h_ls_data))
            t_ber_mmse += sys.calculate_ber(x_bits_data, sys.demodulate_qpsk(y_rx[sys.data_indices] / h_mmse_data))
            t_ber_lml += sys.calculate_ber(x_bits_data, sys.demodulate_qpsk(y_rx[sys.data_indices] / h_lml_data))

        # Average
        results_nmse['ls'].append(t_nmse_ls / num_monte_carlo)
        results_nmse['mmse'].append(t_nmse_mmse / num_monte_carlo)
        results_nmse['lml'].append(t_nmse_lml / num_monte_carlo)
        
        results_ber['ls'].append(t_ber_ls / num_monte_carlo)
        results_ber['mmse'].append(t_ber_mmse / num_monte_carlo)
        results_ber['lml'].append(t_ber_lml / num_monte_carlo)
        
        print(f"  Eb/N0 {ebn0_db}dB (SNR {snr_db:.2f}dB) | NMSE(LML): {results_nmse['lml'][-1]:.4f} | BER(LML): {results_ber['lml'][-1]:.4f}")

    return ebn0_db_list, results_nmse, results_ber

def run_simulation_fig11():
    sys = OFDMSystem()
    
    # Eb/N0 setting
    ebn0_range = np.arange(0, 30, 5) 
    snr_range = ebn0_range + 3.01 # QPSK (2 bits) adjustment
    num_monte_carlo = 200 # 可依需求調整

    ber_proposed = []
    ber_dnn = []
    ber_elm = []
    ber_mmse = []

    print(f"Starting Fig. 11 Simulation on {device}...")

    # 初始化 Estimators
    lml_prop = LMLestimator(sys)
    mmse_est = MMSE(sys)

    for idx, snr_db in enumerate(snr_range):
        snr_lin = 10**(snr_db/10)
        mmse_est.calculate_weights(snr_lin) # DU-MMSE

        err_prop, err_dnn, err_elm, err_mmse = 0, 0, 0, 0
        
        print(f"  Processing Eb/N0 = {ebn0_range[idx]} dB")

        for mc in range(num_monte_carlo):
            # ---------------------------
            # 1. Training Phase (Online)
            # ---------------------------
            H_true = sys.generate_channel()
            
            # Pilot Transmission
            x_pilot, _ = sys.qpsk_modulation(sys.K_active)
            y_clean = H_true * x_pilot
            
            # Apply Non-linear Distortion
            y_distorted = sys.nonlinear_distortion(y_clean)
            
            # Add Noise
            y_rx, _ = sys.add_noise(y_distorted, snr_db)
            
            # LS Estimation (Noisy Labels)
            h_ls = y_rx / x_pilot
            
            # Prepare Training Data
            X_train_list, Y_train_list = [], []
            window_size = (sys.M - 1) * sys.Df + 1
            for i in range(sys.K_active - window_size + 1):
                vec = h_ls[i : i + window_size]
                X_train_list.append(vec[[0, -1]])
                Y_train_list.append(vec[1:-1])
            
            X_np = np.array(X_train_list).T
            Y_np = np.array(Y_train_list).T

            # --- Train Proposed ---
            lml_prop.train_with_data(X_np, Y_np)

            # --- Train ELM ---
            elm = ELM_Estimator(input_dim=sys.M, hidden_dim=8, output_dim=sys.S)
            elm.train(X_np, Y_np)

            # --- Train DNN ---
            dnn = ChannelNet(input_dim=2*sys.M, output_dim=2*sys.S).to(device)
            optimizer = optim.Adam(dnn.parameters(), lr=0.01)
            criterion = nn.MSELoss()
            
            inputs_torch = complex_to_real_torch(X_np)
            labels_torch = complex_to_real_torch(Y_np)
            
            # 2. 數據標準化 (Standardization) - 非常關鍵！
            # 計算 training set 的 mean 和 std
            mean_in = inputs_torch.mean(dim=0)
            std_in = inputs_torch.std(dim=0) + 1e-8 # 避免除以 0
            
            mean_out = labels_torch.mean(dim=0)
            std_out = labels_torch.std(dim=0) + 1e-8

            # 正規化輸入與標籤
            inputs_norm = (inputs_torch - mean_in) / std_in
            labels_norm = (labels_torch - mean_out) / std_out

            # 3. 初始化模型
            dnn = ChannelNet(input_dim=2*sys.M, output_dim=2*sys.S).to(device)
            
            # 4. 優化器設定
            # lr: 降低到 0.005 或 0.001 以求穩定
            # weight_decay: 加入 L2 正則化 (1e-4) 防止過擬合雜訊
            optimizer = optim.Adam(dnn.parameters(), lr=0.005, weight_decay=1e-4)
            criterion = nn.MSELoss()
            
            # 5. 訓練迴圈
            dnn.train()
            # 增加 epochs 數 (例如 100~200)，因為加入了正規化需要更多時間收斂
            for epoch in range(150): 
                optimizer.zero_grad()
                out = dnn(inputs_norm) # 使用正規化後的輸入
                loss = criterion(out, labels_norm) # 與正規化後的標籤比較
                loss.backward()
                optimizer.step()

            # ---------------------------
            # 2. Testing Phase (Data)
            # ---------------------------
            x_data, x_ints = sys.qpsk_modulation(sys.K_active)
            y_data_clean = H_true * x_data
            y_data_distorted = sys.nonlinear_distortion(y_data_clean)
            y_data_rx, _ = sys.add_noise(y_data_distorted, snr_db)

            # Prepare Inputs (LS at pilots)
            input_vectors = []
            for g in range(sys.num_groups):
                p1 = y_data_rx[sys.pilot_indices[g]] / x_data[sys.pilot_indices[g]]
                p2 = y_data_rx[sys.pilot_indices[g+1]] / x_data[sys.pilot_indices[g+1]]
                input_vectors.append(np.array([p1, p2]))
            input_block = np.array(input_vectors).T

            # Infer
            # Proposed (Recalculate simple LS at pilots for LML standard input)
            ls_pilots_vec = np.zeros(len(sys.pilot_indices), dtype=complex)
            for k_idx, k_val in enumerate(sys.pilot_indices):
                ls_pilots_vec[k_idx] = y_data_rx[k_val] / x_data[k_val]
            
            h_est_prop = lml_prop.estimate(ls_pilots_vec)
            h_est_mmse = mmse_est.estimate(ls_pilots_vec)
            
            # ELM Infer
            out_elm = elm.predict(input_block)
            h_est_elm = out_elm.flatten('F')

            # DNN Infer
            dnn.eval()
            with torch.no_grad():
                inp_torch = complex_to_real_torch(input_block)
                out_dnn_torch = dnn(inp_torch)
                out_dnn = real_torch_to_complex(out_dnn_torch)
            h_est_dnn = out_dnn.flatten('F')

            # ---------------------------
            # 3. Detection & BER
            # ---------------------------
            data_indices_all = []
            for g in range(sys.num_groups):
                base = sys.pilot_indices[g]
                data_indices_all.extend([base+1, base+2])
            
            y_eval = y_data_rx[data_indices_all]
            x_ints_eval = x_ints[data_indices_all]

            def get_ber(y, h, true_bits):
                x_hat = y / h
                dec_bits = sys.demodulate_qpsk(x_hat)
                return sys.calculate_ber(true_bits, dec_bits)

            err_prop += get_ber(y_eval, h_est_prop, x_ints_eval)
            err_mmse += get_ber(y_eval, h_est_mmse, x_ints_eval)
            err_elm += get_ber(y_eval, h_est_elm, x_ints_eval)
            err_dnn += get_ber(y_eval, h_est_dnn, x_ints_eval)

        # Average BER
        ber_proposed.append(err_prop / num_monte_carlo)
        ber_mmse.append(err_mmse / num_monte_carlo)
        ber_elm.append(err_elm / num_monte_carlo)
        ber_dnn.append(err_dnn / num_monte_carlo)

    return ebn0_range, ber_proposed, ber_dnn, ber_elm, ber_mmse
