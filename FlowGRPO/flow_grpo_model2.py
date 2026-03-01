import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ===================== 1. 全局参数配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42  # 随机种子，确保实验可复现
np.random.seed(SEED)
torch.manual_seed(SEED)

# 核心参数配置
X0_SAMPLES = np.array([-2.0, 2.0])  # 目标数据分布：两个点分别在-2和2
NUM_TRAIN_SAMPLES = 100000  # 训练样本数量
EPOCHS = 120  # 训练轮数
BATCH_SIZE = 128  # 批次大小
LR = 1e-4  # 学习率
WEIGHT_DECAY = 1e-5  # 权重衰减（L2正则化）
MODEL_SAVE_PATH = "flow_matching_model2.pth" # 模型保存路径

# SDE和可视化参数
T = 1.0  # 最大时间步
NUM_STEPS = 100  # 时间步数
DT = T / NUM_STEPS  # 时间间隔
NUM_SAMPLES = 1 # 采样的样本数
A = 0.5  # SDE噪声水平
T_LIST = np.linspace(0, T, NUM_STEPS)  # 时间点列表
STEP_INDICES = np.arange(NUM_STEPS)  # 时间步索引
x1_test = 0.0  # 子图2的起始噪声点位置

# ===================== 2. 训练集构造 =====================
def generate_flow_matching_data(num_samples, x0_candidates):
    """
    生成Flow Matching训练数据
    :param num_samples: 生成的样本数量
    :param x0_candidates: 目标数据分布的候选均值
    :return: 输入张量、目标张量、归一化参数
    """
    # 1. 生成原始数据 - 使用Flow Matching的核心思想
    # 先随机选择一个候选均值(-2.0或2.0)
    x0_means = np.random.choice(x0_candidates, size=num_samples)
    # 以该均值为中心，添加高斯噪声(方差为1)
    x0 = np.random.normal(loc=x0_means, scale=0.1, size=num_samples)
    x1 = np.random.normal(0, 1, size=num_samples)  # 从标准正态分布采样x1（噪声）
    print(f"x1 max: {x1.max():.4f}, min: {x1.min():.4f}")
    print(f"x0 max: {x0.max():.4f}, min: {x0.min():.4f}, mean: {x0.mean():.4f}")
    t = np.random.uniform(0, 1, size=num_samples)  # 从[0,1]均匀采样时间步
    # t = np.random.beta(1.0, 3.0, size=num_samples)  # Beta分布，偏向小t值，增加小t采样
    # 线性插值路径：x_t = (1-t)x0 + tx1
    x_t = (1 - t) * x0 + t * x1
    # 真实向量场：v_t = x1 - x0
    v_t_gt = x1 - x0

    # 2. 转换为张量（无归一化）
    inputs = torch.tensor(np.stack([x_t, t], axis=1), dtype=torch.float32).to(DEVICE)
    targets = torch.tensor(v_t_gt, dtype=torch.float32).to(DEVICE).unsqueeze(1)
    
    # 返回无归一化的数据 + 占位符归一化参数
    return inputs, targets, (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)



# ===================== 3. Flow Matching MLP模型 =====================
class FlowMatchingMLP(nn.Module):
    """
    Flow Matching向量场模型
    输入：(x_t, t) - 当前状态和时间步
    输出：v_t - 向量场，预测从x_t到x1的移动方向
    """
    def __init__(self, hidden_dim=128, num_hidden_layers=2, norm_params=None):
        super().__init__()
        # 构建MLP网络结构
        layers = [
            nn.Linear(2, hidden_dim),  # 输入层：2维输入(x_t, t)
            nn.ReLU(),
            nn.Dropout(0.1)  # Dropout正则化，防止过拟合
        ]
        # 添加隐藏层
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
        layers.append(nn.Linear(hidden_dim, 1))  # 输出层：1维输出v_t
        self.mlp = nn.Sequential(*layers)
        # 保存归一化参数（推理时用）
        self.norm_params = norm_params

    def forward(self, x_t, t):
        """
        前向传播函数
        :param x_t: 当前状态
        :param t: 当前时间步
        :return: 预测的向量场v_t
        """
        # 处理各种输入类型，转换为张量
        if isinstance(x_t, (int, float, np.ndarray)):
            x_t = torch.tensor(x_t, dtype=torch.float32).to(DEVICE)
        if isinstance(t, (int, float, np.ndarray)):
            t = torch.tensor(t, dtype=torch.float32).to(DEVICE)
        
        # 确保在正确的设备上
        if x_t.device != DEVICE:
            x_t = x_t.to(DEVICE)
        if t.device != DEVICE:
            t = t.to(DEVICE)

        # 形状处理
        if x_t.dim() == 0:
            x_t = x_t.unsqueeze(0)
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        if x_t.shape[-1] == 1 and x_t.dim() > 1:
            x_t = x_t.squeeze(-1)

        if t.shape != x_t.shape:
            t = t.expand_as(x_t)

        # 将x_t和t拼接为输入
        inputs = torch.stack([x_t, t], dim=-1)
        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)

        # 前向传播（无归一化）
        v_t = self.mlp(inputs)

        return v_t.squeeze(-1)

# ===================== 4. Flow Matching训练函数 =====================
def train_flow_matching_model(model, inputs, targets, epochs, batch_size, lr, weight_decay):
    """
    训练Flow Matching模型
    :param model: 待训练的模型
    :param inputs: 输入数据
    :param targets: 目标数据
    :param epochs: 训练轮数
    :param batch_size: 批次大小
    :param lr: 学习率
    :param weight_decay: 权重衰减
    :return: 训练好的模型
    """
    # 初始化优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()  # 使用均方误差损失
    dataset = torch.utils.data.TensorDataset(inputs, targets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_history = []
    model.train()
    for epoch in tqdm(range(epochs), desc="Training"):
        total_loss = 0.0
        for batch_inputs, batch_targets in dataloader:
            optimizer.zero_grad()
            outputs = model.mlp(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_inputs.shape[0]
        
        avg_loss = total_loss / len(dataset)
        loss_history.append(avg_loss)
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
    
    # 绘制损失曲线
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history, 'b-', label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Loss Curve (Should Decrease!)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    return model

# ===================== 5. SDE采样相关函数 =====================
def sigma_t(t):
    """
    SDE的扩散系数函数
    :param t: 时间步
    :return: 扩散系数sigma(t)
    """
    eps = 1e-6
    return A * np.sqrt(t / (1 - t + eps))

def drift_correction(x_t, t, model):
    """
    SDE的漂移修正项（用于修正ODE路径）
    :param x_t: 当前状态
    :param t: 时间步
    :param model: Flow Matching模型
    :return: 漂移修正项
    """
    eps = 1e-6
    v_t = model(x_t, t)
    return (sigma_t(t)**2 / (2 * (t + eps))) * (x_t + (1 - t) * v_t)

def compute_reverse_sde(model, num_samples, x1=0.0):
    """
    计算反向SDE采样（从噪声到数据）
    :param model: Flow Matching模型
    :param num_samples: 采样的样本数
    :param x1: 初始噪声点
    :return: 采样路径及各组成部分
    """
    reverse_sde_samples = []
    reverse_sde_vt_drift = []
    reverse_sde_correction_drift = []
    reverse_sde_diffusion = []
    reverse_sde_vt_step = []
    reverse_sde_correction_step = []
    reverse_sde_diffusion_step = []

    model.eval()
    with torch.no_grad():
        for _ in range(num_samples):
            x = np.zeros(NUM_STEPS)
            x[-1] = x1
            vt_contrib = np.zeros(NUM_STEPS)
            corr_contrib = np.zeros(NUM_STEPS)
            diff_contrib = np.zeros(NUM_STEPS)
            vt_step = np.zeros(NUM_STEPS)
            corr_step = np.zeros(NUM_STEPS)
            diff_step = np.zeros(NUM_STEPS)

            for i in range(NUM_STEPS-2, -1, -1): 
                t = T_LIST[i]
                current_vt = model(x[i+1], t).item() if hasattr(model(x[i+1], t), 'item') else model(x[i+1], t)
                vt_drift = float(current_vt) * -DT # 关键修复-DT处理！
                corr_drift = drift_correction(x[i+1], t, model) * -DT
                diffusion = sigma_t(t) * np.random.normal(0, 1) * np.sqrt(DT)

                x[i] = x[i+1] + vt_drift + corr_drift + diffusion
                vt_contrib[i] = vt_contrib[i+1] + vt_drift
                corr_contrib[i] = corr_contrib[i+1] + corr_drift
                diff_contrib[i] = diff_contrib[i+1] + diffusion
                vt_step[i] = vt_drift
                corr_step[i] = corr_drift
                diff_step[i] = diffusion

            reverse_sde_samples.append(x)
            reverse_sde_vt_drift.append(vt_contrib)
            reverse_sde_correction_drift.append(corr_contrib)
            reverse_sde_diffusion.append(diff_contrib)
            reverse_sde_vt_step.append(vt_step)
            reverse_sde_correction_step.append(corr_step)
            reverse_sde_diffusion_step.append(diff_step)

    return (reverse_sde_samples, reverse_sde_vt_drift, reverse_sde_correction_drift,
            reverse_sde_diffusion, reverse_sde_vt_step, reverse_sde_correction_step,
            reverse_sde_diffusion_step)

# ===================== 6. 模型可视化函数 =====================
def visualize_model(model):
    reverse_sde_results = compute_reverse_sde(model, NUM_SAMPLES, x1=x1_test)
    (reverse_sde_samples, reverse_sde_vt_drift, reverse_sde_correction_drift,
     reverse_sde_diffusion, reverse_sde_vt_step, reverse_sde_correction_step,
     reverse_sde_diffusion_step) = reverse_sde_results

    plt.rcParams['font.size'] = 10
    fig = plt.figure(figsize=(20, 16))

    # 子图1：ODE vs SDE反向过程
    ax1 = plt.subplot(3, 2, 1)
    reverse_ode_x = np.zeros(NUM_STEPS)
    reverse_ode_x[-1] = 0.0
    for i in range(NUM_STEPS-2, -1, -1):
        t = T_LIST[i]
        current_vt = model(reverse_ode_x[i+1], t).item() if hasattr(model(reverse_ode_x[i+1], t), 'item') else model(reverse_ode_x[i+1], t)
        reverse_ode_x[i] = reverse_ode_x[i+1] - current_vt * DT

    ax1.plot(T_LIST, reverse_ode_x, 'r-', linewidth=2, label='Reverse ODE (Model)')
    for i, sample in enumerate(reverse_sde_samples):
        ax1.plot(T_LIST, sample, 'b--', alpha=0.5, label=f'Sample {i+1}' if i==0 else "")
    ax1.set_xlabel('t'); ax1.set_ylabel('x_t')
    ax1.set_title('Reverse Process: ODE vs SDE')
    ax1.legend(); ax1.grid(alpha=0.3)

    # 子图2：SDE各组成部分
    ax2 = plt.subplot(3, 2, 2)
    x1_val = x1_test
    for i in range(NUM_SAMPLES):
        ax2.plot(T_LIST, reverse_sde_samples[i], 'k-', alpha=0.8, label=f'SDE {i+1}' if i==0 else "")
        ax2.plot(T_LIST, x1_val + reverse_sde_vt_drift[i], 'r--', alpha=0.6, label=f'v_t {i+1}' if i==0 else "")
        ax2.plot(T_LIST, x1_val + reverse_sde_correction_drift[i], 'b:', alpha=0.6, label=f'Corr {i+1}' if i==0 else "")
        ax2.plot(T_LIST, x1_val + reverse_sde_diffusion[i], 'g-.', alpha=0.6, label=f'Diff {i+1}' if i==0 else "")
    ax2.set_xlabel('t'); ax2.set_ylabel('x_t / Cumul.')
    ax2.set_title('Reverse SDE: v_t / Correction / Diffusion'); ax2.legend(); ax2.grid(alpha=0.3)

    # 子图3：SDE逐步分解
    ax3 = plt.subplot(3, 2, 3)
    for i in range(NUM_SAMPLES):
        ax3.plot(STEP_INDICES, reverse_sde_vt_step[i], 'r--', alpha=0.6, label=f'v_t step {i+1}' if i==0 else "")
        ax3.plot(STEP_INDICES, reverse_sde_correction_step[i], 'b:', alpha=0.6, label=f'Corr step {i+1}' if i==0 else "")
        ax3.plot(STEP_INDICES, reverse_sde_diffusion_step[i], 'g-.', alpha=0.6, label=f'Diff step {i+1}' if i==0 else "")
    ax3.set_xlabel('step'); ax3.set_ylabel('step increment')
    ax3.set_title('Reverse SDE Stepwise'); ax3.legend(); ax3.grid(alpha=0.3)

    # 子图4：多初始噪声的ODE路径
    ax4 = plt.subplot(3, 2, 4)
    num_samples = 100
    initial_x1 = np.linspace(-4, 4, num_samples) 

    model.eval()
    with torch.no_grad():
        for x1_val in initial_x1:
            reverse_ode_x = np.zeros(NUM_STEPS)
            reverse_ode_x[-1] = x1_val
            for i in range(NUM_STEPS-2, -1, -1):
                t = T_LIST[i]
                current_vt = model(reverse_ode_x[i+1], t).item() if hasattr(model(reverse_ode_x[i+1], t), 'item') else model(reverse_ode_x[i+1], t)
                reverse_ode_x[i] = reverse_ode_x[i+1] - current_vt * DT
            ax4.plot(T_LIST, reverse_ode_x, 'b--', alpha=0.5)

    # 标记两个均值点
    ax4.scatter([0, 0], [-2, 2], c='red', s=100, marker='o', label='Target Means (-2, 2)')
    ax4.set_xlabel('t')
    ax4.set_ylabel('x_t')
    ax4.set_title('Reverse ODE with Multiple Initial Noise Samples\n(Target: Gaussian around -2 and 2)')
    ax4.legend()
    ax4.grid(alpha=0.3)

    # 子图5：固定t的向量场
    ax5 = plt.subplot(3, 2, 5)
    x_vals = np.linspace(-5, 5, 200)
    for t_val in [0.1, 0.5, 0.9]:
        v_vals = [model(x, t_val).item() if hasattr(model(x, t_val), 'item') else model(x, t_val) for x in x_vals]
        ax5.plot(x_vals, v_vals, label=f't={t_val}')
    ax5.axvline(-2, c='r', ls='--', alpha=0.5, label='Mean -2')
    ax5.axvline(2, c='r', ls='--', alpha=0.5, label='Mean 2')
    ax5.set_xlabel('x'); ax5.set_ylabel('v_t(x)')
    ax5.set_title('v_t(x) at fixed t'); ax5.legend(); ax5.grid(alpha=0.3)

    # 子图6：固定x的向量场
    ax6 = plt.subplot(3, 2, 6)
    t_vals = np.linspace(0, 1, 100)
    for x_val in [-3.0, -2.0, 0.0, 2.0, 3.0]:
        v_vals = [model(x_val, t).item() if hasattr(model(x_val, t), 'item') else model(x_val, t) for t in t_vals]
        ax6.plot(t_vals, v_vals, label=f'x={x_val}')
    ax6.set_xlabel('t'); ax6.set_ylabel('v_t(x)')
    ax6.set_title('v_t(x) at fixed x'); ax6.legend(); ax6.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

# ===================== 7. 主程序入口 =====================
if __name__ == "__main__":
    """
    Flow Matching模型的主程序
    - 如果模型已存在，则加载模型
    - 如果模型不存在，则训练模型
    """
    import os
    
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading pre-trained model from {MODEL_SAVE_PATH}...")
        checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE, weights_only=False)
        model = FlowMatchingMLP(hidden_dim=128, num_hidden_layers=2, norm_params=checkpoint['norm_params']).to(DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    else:
        print("No pre-trained model found. Starting training...")
        train_inputs, train_targets, norm_params = generate_flow_matching_data(NUM_TRAIN_SAMPLES, X0_SAMPLES)
        model = FlowMatchingMLP(hidden_dim=128, num_hidden_layers=2, norm_params=norm_params).to(DEVICE)
        model = train_flow_matching_model(model, train_inputs, train_targets, EPOCHS, BATCH_SIZE, LR, WEIGHT_DECAY)
        
        print(f"Saving model to {MODEL_SAVE_PATH}...")
        torch.save({
            'model_state_dict': model.state_dict(),
            'norm_params': model.norm_params
        }, MODEL_SAVE_PATH)
        print("Model saved successfully!")
    
    visualize_model(model)
