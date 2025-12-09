import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# 引用您建立的模組
from OFDMsystem import OFDMSystem
from LMLestimator import LML_Estimator
from MMSE import MMSE_Estimator

def run_simulation():
    # ==========================================
    # 1. 參數設定 (Parameter Configuration)
    # ==========================================
    # 系統參數
    K = 64                  # 子載波總數
    CP = 16                 # 循環前綴
    Df = 4                  # 導頻間隔 (Pilot Spacing)
    M = 2                   # LML 分組導頻數 (Group Size)
    
    # 模擬參數
    SNR_dB_range = [0, 5, 10, 15, 20, 25, 30]  # 測試的 SNR 範圍
    num_iterations = 1000   # 每個 SNR 跑多少次蒙地卡羅模擬
    MMSE_sample = 1000
    
    # 結果儲存容器
    nmse_results = {
        'LS_Linear': [], # 傳統線性插值
        'LML': [],        # 論文提出的方法
        'MMSE': []     # 預留給 MMSE
    }
    
    ber_results = {
        'LS_Linear': [],
        'LML': [],
        'MMSE': []
    }

    print(f"開始模擬... K={K}, Iterations={num_iterations}")
    print("-" * 50)

    # ==========================================
    # 2. 模擬主循環 (Main Loop)
    # ==========================================
    for snr in SNR_dB_range:
        print(f"Processing SNR = {snr} dB ...")
        
        # 累積誤差統計
        mse_accum = {'LS_Linear': 0.0, 'LML': 0.0, 'MMSE':0.0}
        bit_error_accum = {'LS_Linear': 0, 'LML': 0, 'MMSE':0}
        total_bits = 0
        
        for i in range(num_iterations):
            # -----------------------------------
            # A. 環境初始化
            # -----------------------------------
            sys = OFDMSystem(K=K, CP=CP, pilot_spacing=Df, channel_taps=8)
            lml = LML_Estimator(M=M, pilot_spacing=Df, test=False)
            mmse_est = MMSE_Estimator(sys, n_samples=MMSE_sample, test=False)
            
            # 生成真實信道 (假設 Block Fading，訓練和數據階段保持不變)
            h_t, h_f_true = sys.generate_random_channel()
            
            # -----------------------------------
            # B. 階段一：線上訓練 (Online Training)
            # -----------------------------------
            # 發送 Block Pilot (全導頻)
            tx_train, x_train_f = sys.transmitter(payload_bits=[], mode='training')
            # 注意：訓練階段通常 SNR 較高，或者假設接收端能通過多次平均獲得較準確的估計
            # 這裡簡單起見，使用當前測試的 SNR (或者您可以設為固定高 SNR)
            rx_train = sys.channel_propagation(tx_train, h_t, snr_db=snr) 
            y_train_f = sys.receiver_processing(rx_train)
            
            # 獲取全頻帶 LS 估計 (有噪聲)
            h_block_ls = y_train_f / x_train_f
            
            # 訓練 LML 模型
            lml.train(h_block_ls)
            
            # -----------------------------------
            # C. 階段二：數據傳輸 (Data Transmission)
            # -----------------------------------
            # 生成隨機數據位元
            n_data_bits = sys.n_data * 2 # QPSK = 2 bits
            bits_tx = np.random.randint(0, 2, n_data_bits)
            
            # 發送梳狀導頻信號
            tx_data, x_data_f = sys.transmitter(bits_tx, mode='data')
            rx_data = sys.channel_propagation(tx_data, h_t, snr_db=snr)
            y_data_f = sys.receiver_processing(rx_data)
            
            # 提取導頻位置的 LS 估計
            # 假設接收端已知導頻序列 (這裡是 1+0j)
            pilot_rx = y_data_f[sys.pilot_indices]
            pilot_tx = x_data_f[sys.pilot_indices]
            h_pilot_ls = pilot_rx / pilot_tx
            
            # -----------------------------------
            # D. 信道估計算法比較
            # -----------------------------------
            
            # 方法 1: 傳統線性插值 (LS + Linear Interp)
            # 注意：interp1d 預設不處理循環邊界，這裡用 extrapolate 簡單處理
            f_interp = interp1d(sys.pilot_indices, h_pilot_ls, kind='linear', fill_value="extrapolate")
            h_est_linear = f_interp(np.arange(K))
            
            # 方法 2: LML (Proposed)
            # 務必傳入 K 以處理循環邊界
            h_est_lml = lml.estimate(h_pilot_ls, K_total=K)
            
            # [預留空間] 方法 3: MMSE
            h_est_mmse = mmse_est.estimate(h_pilot_ls, snr_db=snr)
            
            # -----------------------------------
            # E. 性能評估 (NMSE & BER)
            # -----------------------------------
            
            # 1. 計算 NMSE
            power_h = np.sum(np.abs(h_f_true)**2)
            
            mse_accum['LS_Linear'] += np.sum(np.abs(h_f_true - h_est_linear)**2) / power_h
            mse_accum['LML'] += np.sum(np.abs(h_f_true - h_est_lml)**2) / power_h
            mse_accum['MMSE'] += np.sum(np.abs(h_f_true - h_est_mmse)**2) / power_h
            
            # 2. 計算 BER (需要進行均衡 Equalization)
            # 使用 Zero-Forcing Equalizer: Y_data / H_est
            
            # 提取真實接收到的數據位置信號
            y_data_subcarriers = y_data_f[sys.data_indices]
            
            # 定義一個簡單的解調與BER計算函數
            def calc_ber(h_est, y_received, bits_true):
                h_data_est = h_est[sys.data_indices]
                # ZF Equalization
                sym_est = y_received / h_data_est
                # QPSK Demodulation (簡單象限判斷)
                bits_est = []
                for s in sym_est:
                    # QPSK: Re>0 -> 0, Re<0 -> 1; Im>0 -> 0, Im<0 -> 1 (視您的映射規則而定)
                    # 對應 OFDMSystem 的映射: 0->1/sq2, 1->-1/sq2
                    b0 = 0 if s.real > 0 else 1
                    b1 = 0 if s.imag > 0 else 1
                    bits_est.extend([b0, b1])
                return np.sum(np.abs(np.array(bits_true) - np.array(bits_est)))

            bit_error_accum['LS_Linear'] += calc_ber(h_est_linear, y_data_subcarriers, bits_tx)
            bit_error_accum['LML'] += calc_ber(h_est_lml, y_data_subcarriers, bits_tx)
            bit_error_accum['MMSE'] += calc_ber(h_est_mmse, y_data_subcarriers, bits_tx)
            
            total_bits += n_data_bits

        # -----------------------------------
        # 計算平均並儲存
        # -----------------------------------
        # NMSE (dB) = 10 * log10(Mean MSE)
        nmse_results['LS_Linear'].append(10 * np.log10(mse_accum['LS_Linear'] / num_iterations))
        nmse_results['LML'].append(10 * np.log10(mse_accum['LML'] / num_iterations))
        nmse_results['MMSE'].append(10 * np.log10(mse_accum['MMSE'] / num_iterations))
        
        # BER
        ber_results['LS_Linear'].append(bit_error_accum['LS_Linear'] / total_bits)
        ber_results['LML'].append(bit_error_accum['LML'] / total_bits)
        ber_results['MMSE'].append(bit_error_accum['MMSE'] / total_bits)

    # ==========================================
    # 3. 繪圖 (Plotting)
    # ==========================================
    plot_results(SNR_dB_range, nmse_results, ber_results)

def plot_results(snr_range, nmse_data, ber_data):
    # Plot NMSE
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(snr_range, nmse_data['LS_Linear'], 'b--o', label='LS + Linear Interp')
    plt.plot(snr_range, nmse_data['LML'], 'r-s', label='LML (Proposed)')
    # plt.plot(snr_range, nmse_data['MMSE'], 'g-^', label='MMSE') # 預留
    plt.plot(snr_range, nmse_data['MMSE'], 'g-^', label='MMSE (Ideal)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('NMSE (dB)')
    plt.title('Channel Estimation NMSE')
    plt.grid(True)
    plt.legend()
    
    # Plot BER
    plt.subplot(1, 2, 2)
    plt.semilogy(snr_range, ber_data['LS_Linear'], 'b--o', label='LS + Linear Interp')
    plt.semilogy(snr_range, ber_data['LML'], 'r-s', label='LML (Proposed)')
    plt.semilogy(snr_range, ber_data['MMSE'], 'g-^', label='MMSE (Ideal)')
    plt.xlabel('SNR (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('System BER (QPSK)')
    plt.grid(True, which="both")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()