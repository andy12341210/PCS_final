import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 引用您的自定義模組
from OFDMsystem import OFDMSystem
from LMLestimator import LMLestimator
from MMSE import MMSE
from Neural_estimators import ChannelNet, ELM_Estimator, complex_to_real_torch, real_torch_to_complex

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            
            dnn.train()
            for epoch in range(50):
                optimizer.zero_grad()
                out = dnn(inputs_torch)
                loss = criterion(out, labels_torch)
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

def run_simulation_fig5():
    # 初始化物件
    ofdm = OFDMSystem()
    mmse_est = MMSE(ofdm)
    lml_est = LMLestimator(ofdm)

    snr_db_range = np.arange(-10, 20, 4)
    num_monte_carlo = 1000

    nmse_results = {
        'ls': [],
        'mmse': [],
        'proposed': []
    }

    print(f"Starting Figure 5 Simulation (SNR range: {snr_db_range} dB)...")

    for snr_db in snr_db_range:
        snr_lin = 10**(snr_db/10)
        loss_ls = 0
        loss_mmse = 0
        loss_prop = 0

        mmse_est.calculate_weights(snr_lin)

        for mc in range(num_monte_carlo):
            H_true = ofdm.generate_channel()

            # Phase 1: Training
            # [修正] qpsk_modulation 回傳 (symbols, ints)，這裡只取 symbols
            x_block, _ = ofdm.qpsk_modulation(ofdm.K_active)
            y_block_clean = H_true * x_block
            y_block_rx, _ = ofdm.add_noise(y_block_clean, snr_db)
            h_ls_block = y_block_rx / x_block
            
            lml_est.train(h_ls_block)

            # Phase 2: Testing
            noise_power = 1.0 / snr_lin
            noise_at_pilots = (np.random.randn(len(ofdm.pilot_indices)) + 1j*np.random.randn(len(ofdm.pilot_indices))) / np.sqrt(2)
            h_ls_pilots = H_true[ofdm.pilot_indices] + np.sqrt(noise_power) * noise_at_pilots

            h_true_data = H_true[ofdm.data_indices]

            # LS
            h_est_ls = ofdm.ls_interpolation(h_ls_pilots)
            loss_ls += ofdm.calc_nmse(h_true_data, h_est_ls)

            # MMSE
            h_est_mmse = mmse_est.estimate(h_ls_pilots)
            loss_mmse += ofdm.calc_nmse(h_true_data, h_est_mmse)

            # Proposed
            h_est_prop = lml_est.estimate(h_ls_pilots)
            loss_prop += ofdm.calc_nmse(h_true_data, h_est_prop)

        nmse_results['ls'].append(loss_ls / num_monte_carlo)
        nmse_results['mmse'].append(loss_mmse / num_monte_carlo)
        nmse_results['proposed'].append(loss_prop / num_monte_carlo)

        print(f"  SNR {snr_db}dB | LS: {nmse_results['ls'][-1]:.4f} | MMSE: {nmse_results['mmse'][-1]:.4f} | Prop: {nmse_results['proposed'][-1]:.4f}")

    return snr_db_range, nmse_results