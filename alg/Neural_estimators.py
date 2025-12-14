import numpy as np
import torch
import torch.nn as nn

# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Helper Functions for Complex <-> Real conversion ---
def complex_to_real_torch(data_complex):
    # (Features, Samples) -> (Samples, Features*2)
    real_part = data_complex.real
    imag_part = data_complex.imag
    tensor = np.vstack([real_part, imag_part]).T
    return torch.tensor(tensor, dtype=torch.float32).to(device)

def real_torch_to_complex(data_real):
    # (Samples, 2*Features) -> (Features, Samples)
    data_np = data_real.cpu().detach().numpy().T
    mid = data_np.shape[0] // 2
    return data_np[:mid, :] + 1j * data_np[mid:, :]

# --- DNN Model ---
class ChannelNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ChannelNet, self).__init__()
        # 修改 1: 加寬隱藏層 (例如 32)，並移除瓶頸
        # 修改 2: 使用 Tanh 激活函數，對模擬平滑通道響應效果通常比 ReLU 好
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, output_dim)
        )
        
        # 初始化權重 (Xavier Initialization 對 Tanh 較好)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

# --- ELM Estimator ---
class ELM_Estimator:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # Random input weights (Fixed)
        self.W_in = np.random.randn(hidden_dim, input_dim) + 1j * np.random.randn(hidden_dim, input_dim)
        self.b = np.random.randn(hidden_dim, 1) + 1j * np.random.randn(hidden_dim, 1)
        self.W_out = None

    def activation(self, x):
        # Activation for complex numbers
        return np.tanh(x.real) + 1j * np.tanh(x.imag)

    def train(self, X, Y):
        # X: (Input_Dim, N), Y: (Output_Dim, N)
        H = self.activation(np.dot(self.W_in, X) + self.b) 
        H_pinv = np.linalg.pinv(H)
        self.W_out = np.dot(Y, H_pinv)

    def predict(self, X):
        H = self.activation(np.dot(self.W_in, X) + self.b)
        return np.dot(self.W_out, H)