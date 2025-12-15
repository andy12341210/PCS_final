import matplotlib.pyplot as plt
import numpy as np
import os
from sim import run_simulation_fig6, run_simulation_fig11, run_simulation_fig5

# 設定儲存路徑
RESULT_DIR = './result'
if not os.path.exists(RESULT_DIR):
    os.makedirs(RESULT_DIR)
    print(f"Created directory: {RESULT_DIR}")

def plot_fig5():
    x, res_nmse, res_ber = run_simulation_fig5()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: NMSE ---
    ax1.semilogy(x, res_nmse['ls'], 'k-s', label='LS')
    ax1.semilogy(x, res_nmse['proposed'], 'r-v', label='Proposed (PATDG)')
    ax1.semilogy(x, res_nmse['ddtdg'], 'g-o', label='Proposed (DDTDG)') # 新增
    ax1.semilogy(x, res_nmse['mmse'], 'b--*', label='MMSE')

    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('NMSE')
    ax1.grid(True, which="both", ls="-", alpha=0.5)
    ax1.legend()
    ax1.set_title('Figure 5a: NMSE vs SNR')

    # --- Plot 2: BER ---
    ax2.semilogy(x, res_ber['ls'], 'k-s', label='LS')
    ax2.semilogy(x, res_ber['proposed'], 'r-v', label='Proposed (PATDG)')
    ax2.semilogy(x, res_ber['ddtdg'], 'g-o', label='Proposed (DDTDG)') # 新增
    ax2.semilogy(x, res_ber['mmse'], 'b--*', label='MMSE')

    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('BER')
    ax2.grid(True, which="both", ls="-", alpha=0.5)
    ax2.legend()
    ax2.set_title('Figure 5b: BER vs SNR')
    
    plt.tight_layout()
    
    save_path = os.path.join(RESULT_DIR, 'Figure_5_DDTDG.png')
    plt.savefig(save_path)
    print(f"Figure 5 saved to {save_path}")
    plt.close()

def plot_fig6():
    sizes, res_prop, res_acc, res_mmse, snrs = run_simulation_fig6()

    plt.figure(figsize=(10, 8))
    colors = ['r', 'm', 'b'] 
    markers = ['v', '^', 'o']

    for i, snr in enumerate(snrs):
        # Proposed
        plt.semilogy(sizes, res_prop[i], color=colors[i], marker=markers[i], linestyle='-',
                     label=f'Proposed est. SNR={snr} dB')
        # Accurate Labels
        plt.semilogy(sizes, res_acc[i], color=colors[i], marker=markers[i], linestyle='--', alpha=0.6,
                     label=f'Accurate labels SNR={snr} dB')
        # MMSE (Flat line)
        mmse_val = np.mean(res_mmse[i])
        plt.axhline(y=mmse_val, color=colors[i], linestyle='-.', alpha=0.5,
                    label=f'MMSE est. SNR={snr} dB')

    plt.xlabel('Size of the dataset')
    plt.ylabel('NMSE')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('Fig. 6 Reproduction: NMSE vs Dataset Size')
    plt.tight_layout()
    
    save_path = os.path.join(RESULT_DIR, 'Figure_6.png')
    plt.savefig(save_path)
    print(f"Figure 6 saved to {save_path}")
    plt.close()

def plot_fig11():
    x, y_prop, y_dnn, y_elm, y_mmse = run_simulation_fig11()

    plt.figure(figsize=(10, 8))
    plt.semilogy(x, y_dnn, 'k-+', label='DNN based estimation')
    plt.semilogy(x, y_elm, 'k--o', markerfacecolor='none', label='C-ELM based estimation')
    plt.semilogy(x, y_mmse, 'b-x', label='DU-MMSE estimation')
    plt.semilogy(x, y_prop, 'r-v', label='Proposed estimation method')

    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('BER')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    plt.title('Fig. 11 Reproduction: BER vs Eb/N0 (Non-linear Channel)')
    
    save_path = os.path.join(RESULT_DIR, 'Figure_11.png')
    plt.savefig(save_path)
    print(f"Figure 11 saved to {save_path}")
    plt.close()

if __name__ == "__main__":
    
    # Run Fig 5
    plot_fig5()
    
    # Run Fig 6
    plot_fig6()
    
    # Run Fig 11
    plot_fig11()