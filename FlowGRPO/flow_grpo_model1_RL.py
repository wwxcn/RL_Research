import torch
import math
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
import flow_grpo_model1 as fm_model

# ===================== 1. 全局参数配置 =====================
HYPER_PARAMS = {
    "group_size": 24,
    "train_denoise_steps": 30,
    "infer_denoise_steps": 50,
    "noise_level": 0.5,
    "kl_coeff": 0.01,
    "clip_epsilon": 0.2,
    "learning_rate": 1e-4,
    "batch_size": 8,
    "max_epochs": 120,
    "reward_sigma": 0.2
}

DEVICE = fm_model.DEVICE
Tmax = fm_model.T - 0.01
Tmin = 0.01
NUM_STEPS = fm_model.NUM_STEPS # 可视化的时间步长
DT = fm_model.DT
X0_TARGET = 2.0
X0_NEGATIVE = -2.0

# ===================== 2. GRPO初始化函数 =====================
def initialize_grpo(pretrained_model_path):
    """加载预训练模型，初始化GRPO组件"""
    # 1. 加载预训练Flow Matching模型
    checkpoint = torch.load(
        pretrained_model_path, 
        map_location=DEVICE, 
        weights_only=False
    )
    v_theta = fm_model.FlowMatchingMLP(hidden_dim=128, num_hidden_layers=2, norm_params=checkpoint['norm_params']).to(DEVICE)
    v_theta.load_state_dict(checkpoint['model_state_dict'])
    v_theta.norm_params = checkpoint['norm_params']
    # 2. 参考策略（预训练模型，用于KL正则）
    v_ref = copy.deepcopy(v_theta)
    optimizer = optim.Adam(v_theta.parameters(), lr=HYPER_PARAMS["learning_rate"])
    # 3. 预计算训练时间步（t从1.0到0.0）
    T_train = HYPER_PARAMS["train_denoise_steps"]
    timesteps = torch.linspace(Tmax, Tmin, T_train + 1).to(DEVICE)
    delta_t = torch.tensor((Tmax - Tmin) / T_train, dtype=torch.float32).to(DEVICE)
    
    return v_theta, v_ref, optimizer, timesteps, delta_t

# ===================== 3. SDE采样（适配一维场景） =====================
def sde_sample_1d(v_theta, timesteps, delta_t, a):
    # x_t和x_next是策略生成的样本，不需要计算梯度，使用detach()从计算图中分离出张量(评估模式仍然会构建计算图，仍然需要detach)
    T_train = HYPER_PARAMS["train_denoise_steps"]
    # 初始噪声从[-4, 4]范围内均匀采样，或者从标准正态分布采样
    #x_t = (2.0 * torch.rand(1, dtype=torch.float32) - 1.0) * 4.0  # 从[-4.0, 4.0]范围内均匀采样
    x_t = torch.randn(1, dtype=torch.float32) # 从标准正态分布采样x_1 ~ N(0,1)
    x_t = x_t.to(DEVICE).detach()
    trajectory = [x_t]
    # 采样时应该使用评估模式，评估模式可以确保dropout、normalization等训练时的特殊处理被禁用，确保结果的稳定性
    # 确实发现奖励的上升更稳定！！
    v_theta.eval()
    
    for t_idx in range(T_train):
        t = timesteps[t_idx].unsqueeze(0)
        sigma_t = a * torch.sqrt(t / (1 - t + 1e-8))
        
        v = v_theta(x_t, t)
        drift = v + (sigma_t**2 / (2 * (t + 1e-8))) * (x_t + (1 - t) * v)
        # Euler-Maruyama离散化
        epsilon = torch.randn_like(x_t)
        # 关键修复：-delta_t处理！
        x_next = x_t + drift * -delta_t + sigma_t * torch.sqrt(delta_t) * epsilon
        x_next_detach = x_next.detach()
        
        trajectory.append(x_next_detach)
        x_t = x_next_detach  # detach()的作用：从计算图中分离出张量，避免梯度回传
        assert x_next.item()>-10.0 and x_next.item()<10.0, f"x_next out of range: {x_next}"
    
    x_0 = trajectory[-1]
    return trajectory, x_0

# ===================== 4. 奖励函数 =====================
def compute_reward_1d(x0):
    x0 = x0.squeeze() if x0.dim() > 0 else x0
    
    # 正奖励项：靠近目标点X0_TARGET(2.0)时获得高奖励
    # 正奖励项：使用更陡峭的高斯分布reward_sigma=0.2，使样本更集中在2.0附近
    # 奖励权重2.0或者5.0，似乎差别并不大
    dist_to_target = torch.abs(x0 - X0_TARGET)
    reward_pos = 2.0 * torch.exp(-dist_to_target**2 / (2 * HYPER_PARAMS["reward_sigma"]**2))
    
    # 惩罚项1：靠近负样本点X0_NEGATIVE(-2.0)时受到惩罚
    dist_to_negative = torch.abs(x0 - X0_NEGATIVE)
    penalty_neg = 1.0 * torch.exp(-dist_to_negative**2 / (2 * HYPER_PARAMS["reward_sigma"]**2))
    
    # 惩罚项2：当x0不在[-2, 2]区间时，给予指数或者线性惩罚，惩罚随距离增大而增加
    penalty_outside = torch.where(
        torch.abs(x0) > 2.0,
        100.0 * (torch.abs(x0) - 2.0), # 100.0 * torch.exp(2.0 * (torch.abs(x0) - 2.0)),
        torch.tensor(0.0, dtype=torch.float32, device=DEVICE)
    )
    
    # 总奖励 = 正奖励 - 惩罚项1 - 惩罚项2
    total_reward = reward_pos - penalty_neg - penalty_outside
    return total_reward

# ===================== 5. 策略概率比计算（一维） =====================
def compute_policy_ratio_1d(v_theta, v_theta_old, x_t, x_t_prev, t, delta_t, a):
    """
    计算策略概率比 r_t = p_theta(x_{t-1}|x_t) / p_theta_old(x_{t-1}|x_t)
    一维高斯分布：N(μ_theta, σ_t²Δt)，忽略常数项仅保留指数部分
    """
    sigma_t = a * torch.sqrt(t / (1 - t + 1e-8))
    var = sigma_t**2 * delta_t + 1e-8
    # 当前策略均值μ_theta
    v = v_theta(x_t, t)
    mu_theta = x_t + (v + (sigma_t**2 / (2 * (t + 1e-8))) * (x_t + (1 - t) * v)) * -delta_t # 关键修改 -delta_t
    # 旧策略均值μ_theta_old
    v_old = v_theta_old(x_t, t)
    mu_theta_old = x_t + (v_old + (sigma_t**2 / (2 * (t + 1e-8))) * (x_t + (1 - t) * v_old)) * -delta_t
    
    # 高斯分布对数概率密度函数的常数项，维度D=1
    common_value = -0.5 * torch.log(2 * torch.as_tensor(math.pi) * (sigma_t**2 * delta_t))
    # x_t和x_t_prev来自sde_sample_1d函数的采样结果，已经通过detach()与计算图分离
    # 但是log_p_theta需要计算梯度，不能直接从计算图中分离(log_p_theta_old的参数不会被更新，因为copy.deepcopy的操作)
    log_p_theta = -0.5 * (x_t_prev - mu_theta)**2 / var + common_value
    log_p_theta_old = -0.5 * (x_t_prev - mu_theta_old)**2 / var + common_value
    log_ratio = log_p_theta - log_p_theta_old # common_value项被抵消掉
    ratio = torch.exp(log_ratio)
    return ratio

# ===================== 6. GRPO核心训练循环 =====================
def train_grpo(pretrained_model_path):
    """GRPO微调Flow Matching模型"""
    # 初始化组件
    v_theta, v_ref, optimizer, timesteps, delta_t = initialize_grpo(pretrained_model_path)
    G = HYPER_PARAMS["group_size"]
    T_train = HYPER_PARAMS["train_denoise_steps"]
    beta = HYPER_PARAMS["kl_coeff"]
    eps = HYPER_PARAMS["clip_epsilon"]
    a = HYPER_PARAMS["noise_level"]
    
    # 缓存旧策略（初始为预训练模型）
    v_theta_old = copy.deepcopy(v_theta)
    
    for epoch in tqdm(range(HYPER_PARAMS["max_epochs"]), desc="GRPO Finetuning"):
        # 1. 组内采样：生成G条轨迹
        group_trajectories = []
        group_x0 = []
        for _ in range(G):
            trajectory, x0 = sde_sample_1d(v_theta_old, timesteps, delta_t, a)
            group_trajectories.append(trajectory)
            group_x0.append(x0)
        
        # 2. 计算组内奖励
        group_R = torch.stack([compute_reward_1d(x0) for x0 in group_x0]).to(DEVICE)
        
        # 3. 归一化相对优势
        mean_R = group_R.mean()
        std_R = group_R.std() + 1e-8
        group_A = (group_R - mean_R) / std_R
        
        # 4. 遍历时间步计算损失，每个时间步的ratio*归一化相对优势
        step_losses = []
        for t_idx in range(1, T_train + 1): 
            x_t_list = [traj[t_idx - 1].squeeze() for traj in group_trajectories]
            x_t_prev_list = [traj[t_idx].squeeze() for traj in group_trajectories]
            x_t = torch.stack(x_t_list)
            x_t_prev = torch.stack(x_t_prev_list)
            t = timesteps[t_idx - 1]
            # 4.1 策略概率比
            r_t = compute_policy_ratio_1d(v_theta, v_theta_old, x_t, x_t_prev, t, delta_t, a)
            # 4.2 裁剪优势项
            clipped_r_t = torch.clamp(r_t, 1 - eps, 1 + eps)
            advantage_term = torch.min(r_t * group_A, clipped_r_t * group_A)
            # 4.3 KL正则项（约束与参考策略的差异）
            v = v_theta(x_t, t)
            v_ref_val = v_ref(x_t, t)
            sigma_t = a * torch.sqrt(t / (1 - t + 1e-8))
            kl_scalar = (delta_t / 2) * ((sigma_t * (1 - t) / (2 * (t + 1e-8))) + 1 / (sigma_t + 1e-8))**2
            kl_term = kl_scalar * torch.mean((v - v_ref_val)**2)
            # 4.4 单时间步损失（负目标函数，PyTorch最小化）
            step_loss = -torch.mean(advantage_term - beta * kl_term)
            step_losses.append(step_loss)
        # 5. 组损失平均
        group_loss = torch.mean(torch.stack(step_losses))
        # 6. 反向传播
        optimizer.zero_grad()
        # 检查损失值是否有效
        if torch.isnan(group_loss) or torch.isinf(group_loss):
            print(f"Warning: Invalid loss value {group_loss}, skipping this epoch")
        else:
            group_loss.backward()
            # 当梯度范数超过设定的max_norm时，将梯度按比例缩小，确保梯度范数不超过max_norm
            torch.nn.utils.clip_grad_norm_(v_theta.parameters(), max_norm=1.0)
            optimizer.step()
            v_theta_old = copy.deepcopy(v_theta)
        
        if epoch % 10 == 0 and epoch > 0:
            loss_val = group_loss.item() if not torch.isnan(group_loss) and not torch.isinf(group_loss) else float('nan')
            mean_reward = mean_R.item()
            print(f"Epoch {epoch+1}/{HYPER_PARAMS['max_epochs']}, Loss: {loss_val:.6f}, Mean Reward: {mean_reward:.6f}")
    
    return v_theta

# ===================== 7. 可视化函数 =====================
def visualize_before_after(v_pretrained, v_finetuned):
    """对比GRPO微调前后的流场变化（子图1+子图4）"""
    plt.rcParams['font.size'] = 10
    fig = plt.figure(figsize=(20, 8))
    # 子图1：逆向ODE vs SDE（微调前 vs 微调后）
    ax1 = plt.subplot(1, 2, 1)
    # 预训练模型的逆向ODE
    reverse_ode_pretrain = np.zeros(NUM_STEPS, dtype=np.float32)
    reverse_ode_pretrain[-1] = 0.0
    v_pretrained.eval()
    with torch.no_grad():
        for i in range(NUM_STEPS-2, -1, -1):
            t = fm_model.T_LIST[i]
            x_t = torch.tensor(reverse_ode_pretrain[i+1], dtype=torch.float32).to(DEVICE)
            current_vt = v_pretrained(x_t, t).cpu().numpy()
            if current_vt.ndim > 0:
                current_vt = current_vt[0] if current_vt.shape[0] == 1 else current_vt.item()
            reverse_ode_pretrain[i] = reverse_ode_pretrain[i+1] - float(current_vt) * DT
    # 预训练模型的逆向SDE（5个样本）
    pretrain_sde_samples = fm_model.compute_reverse_sde(v_pretrained, 5, x1=0.0)[0]
    
    ax1.plot(fm_model.T_LIST, reverse_ode_pretrain, 'r-', linewidth=2, label='Pretrain ODE')
    for i, sample in enumerate(pretrain_sde_samples):
        ax1.plot(fm_model.T_LIST, sample, 'b--', alpha=0.5, label=f'Pretrain SDE {i+1}' if i==0 else "")
    ax1.set_xlabel('t'); ax1.set_ylabel('x_t')
    ax1.set_title('Pretrained Model: ODE vs SDE')
    ax1.legend(); ax1.grid(alpha=0.3)
    
    ax2 = plt.subplot(1, 2, 2)
    # 微调后模型的逆向ODE
    reverse_ode_finetune = np.zeros(NUM_STEPS, dtype=np.float32)
    reverse_ode_finetune[-1] = 0.0
    v_finetuned.eval()
    with torch.no_grad():
        for i in range(NUM_STEPS-2, -1, -1):
            t = fm_model.T_LIST[i]
            x_t = torch.tensor(reverse_ode_finetune[i+1], dtype=torch.float32).to(DEVICE)
            current_vt = v_finetuned(x_t, t).cpu().numpy()
            if current_vt.ndim > 0:
                current_vt = current_vt[0] if current_vt.shape[0] == 1 else current_vt.item()
            reverse_ode_finetune[i] = reverse_ode_finetune[i+1] - float(current_vt) * DT
    # 预训练模型的逆向ODE（5个样本）
    finetune_sde_samples = fm_model.compute_reverse_sde(v_finetuned, 5, x1=0.0)[0]
    
    ax2.plot(fm_model.T_LIST, reverse_ode_finetune, 'r-', linewidth=2, label='Finetune ODE')
    for i, sample in enumerate(finetune_sde_samples):
        ax2.plot(fm_model.T_LIST, sample, 'b--', alpha=0.5, label=f'Finetune SDE {i+1}' if i==0 else "")
    ax2.set_xlabel('t'); ax2.set_ylabel('x_t')
    ax2.set_title('GRPO Finetuned Model: ODE vs SDE')
    ax2.legend(); ax2.grid(alpha=0.3)
    
    # 子图4：多初始噪声的逆向ODE轨迹（对比）
    fig2 = plt.figure(figsize=(20, 8))
    # 预训练模型：多初始噪声的逆向ODE
    ax3 = plt.subplot(1, 2, 1)
    num_samples = 40
    initial_x1 = np.linspace(-4, 4, num_samples, dtype=np.float32)
    v_pretrained.eval()
    with torch.no_grad():
        for x1_val in initial_x1:
            reverse_ode_x = np.zeros(NUM_STEPS, dtype=np.float32)
            reverse_ode_x[-1] = x1_val
            for i in range(NUM_STEPS-2, -1, -1):
                t = fm_model.T_LIST[i]
                x_t = torch.tensor(reverse_ode_x[i+1], dtype=torch.float32).to(DEVICE)
                current_vt = v_pretrained(x_t, t).cpu().numpy()
                if current_vt.ndim > 0:
                    current_vt = current_vt[0] if current_vt.shape[0] == 1 else current_vt.item()
                reverse_ode_x[i] = reverse_ode_x[i+1] - float(current_vt) * DT
            ax3.plot(fm_model.T_LIST, reverse_ode_x, 'b--', alpha=0.5)
    ax3.scatter([0, 0], [-2, 2], c='black', s=150, marker='*', label='True x0 ∈ {-2,2}')
    ax3.set_xlabel('t'); ax3.set_ylabel('x_t')
    ax3.set_title('Pretrained Model: Multiple Initial Noise')
    ax3.legend(); ax3.grid(alpha=0.3)

    # 微调后模型：多初始噪声的逆向ODE
    ax4 = plt.subplot(1, 2, 2)
    v_finetuned.eval()
    with torch.no_grad():
        for x1_val in initial_x1:
            reverse_ode_x = np.zeros(NUM_STEPS, dtype=np.float32)
            reverse_ode_x[-1] = x1_val
            for i in range(NUM_STEPS-2, -1, -1):
                t = fm_model.T_LIST[i]
                x_t = torch.tensor(reverse_ode_x[i+1], dtype=torch.float32).to(DEVICE)
                current_vt = v_finetuned(x_t, t).cpu().numpy()
                if current_vt.ndim > 0:
                    current_vt = current_vt[0] if current_vt.shape[0] == 1 else current_vt.item()
                reverse_ode_x[i] = reverse_ode_x[i+1] - float(current_vt) * DT
            ax4.plot(fm_model.T_LIST, reverse_ode_x, 'r--', alpha=0.5)
    ax4.scatter([0], [2], c='red', s=200, marker='*', label='Target x0 = 2.0')
    ax4.scatter([0], [-2], c='gray', s=150, marker='*', label='Negative x0 = -2.0')
    ax4.set_xlabel('t'); ax4.set_ylabel('x_t')
    ax4.set_title('GRPO Finetuned Model: Multiple Initial Noise')
    ax4.legend(); ax4.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ===================== 8. 主函数 =====================
if __name__ == "__main__":
    # 预训练模型路径（同flow_grpo_model1.py）
    pretrained_model_path = "flow_matching_model1.pth"
    # 1. 加载预训练模型（用于对比）
    checkpoint = torch.load(
        pretrained_model_path, 
        map_location=DEVICE, 
        weights_only=False
    )
    v_pretrained = fm_model.FlowMatchingMLP(hidden_dim=128, num_hidden_layers=2, norm_params=checkpoint['norm_params']).to(DEVICE)
    v_pretrained.load_state_dict(checkpoint['model_state_dict'])
    v_pretrained.norm_params = checkpoint['norm_params']
    # 2. GRPO微调模型
    v_finetuned = train_grpo(pretrained_model_path)
    # 3. 可视化对比（子图1+子图4）
    visualize_before_after(v_pretrained, v_finetuned)
