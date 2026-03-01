import numpy as np
import matplotlib.pyplot as plt

# ===================== 1. 核心参数设定 =====================
# np.random.seed(42)  # 固定随机种子，保证可复现
x0 = 5.0            # 初始真实数据（一维，方便可视化）
x1 = 0.0            # 最终噪声（标准高斯均值，简化设定）
T = 0.99             # 总时间范围 [0,1]
T0 = 0.01
num_steps = 100     # 离散时间步数
dt = (T - T0) / num_steps  # 时间步长 Δt
num_samples = 10     # SDE 多次采样的次数
a = 0.5             # 扩散系数超参 σ_t = a*sqrt(t/(1-t))
t_list = np.linspace(T0, T, num_steps)  # 时间序列
step_indices = np.arange(num_steps)    # 迭代步索引（子图5/6用）

# ===================== 2. 核心函数定义（修正v_t） =====================
def alpha_t(t):
    """Rectified Flow 线性插值系数 α_t = 1-t"""
    return 1 - t

def beta_t(t):
    """Rectified Flow 线性插值系数 β_t = t"""
    return t

def v_t(x_t, t):
    """
    修正后的理想速度场（正向）：基于Rectified Flow理论推导
    从 x_t = (1-t)x0 + t x1 解出 x1 = (x_t - (1-t)x0)/t，因此 v_t = x1 - x0
    """
    eps = 1e-6  # 避免t→0时除零
    if t < eps:  # t=0时，x_t=x0，v_t=x1-x0（边界条件）
        return x1 - x0
    return (x_t - (1 - t) * x0) / (t + eps) - x0

def sigma_t(t):
    """扩散系数 σ_t = a*sqrt(t/(1-t))，t→1 时加小epsilon避免除零"""
    eps = 1e-6
    return a * np.sqrt(t / (1 - t + eps))

def drift_correction(x_t, t):
    """SDE 漂移项中的修正项：σ_t²/(2t) * (x_t + (1-t)v_t(x_t))"""
    eps = 1e-6
    return (sigma_t(t)**2 / (2 * (t + eps))) * (x_t + (1 - t) * v_t(x_t, t))

# ===================== 3. 正向过程计算（ODE + SDE） =====================
## 3.1 正向 ODE（确定性，使用修正后的v_t）
forward_ode_x = np.zeros(num_steps)
forward_ode_x[0] = x0
for i in range(1, num_steps):
    t = t_list[i]
    current_vt = v_t(forward_ode_x[i-1], t)
    forward_ode_x[i] = forward_ode_x[i-1] + current_vt * dt

## 3.2 正向 SDE（随机，多次采样）
# 累计贡献（子图2用）
forward_sde_samples = []               # 完整轨迹
forward_sde_vt_drift = []              # v_t项累计贡献
forward_sde_correction_drift = []      # 修正项累计贡献
forward_sde_diffusion = []             # 扩散项累计贡献
# 单步增量（子图5用）
forward_sde_vt_step = []               # v_t项单步增量
forward_sde_correction_step = []       # 修正项单步增量
forward_sde_diffusion_step = []        # 扩散项单步增量

for _ in range(num_samples):
    x = np.zeros(num_steps)
    x[0] = x0
    # 累计贡献初始化
    vt_contrib = np.zeros(num_steps)
    corr_contrib = np.zeros(num_steps)
    diff_contrib = np.zeros(num_steps)
    # 单步增量初始化（第0步无增量，设为0）
    vt_step = np.zeros(num_steps)
    corr_step = np.zeros(num_steps)
    diff_step = np.zeros(num_steps)
    
    for i in range(1, num_steps):
        t = t_list[i]
        # 1. 计算单步增量（使用修正后的v_t）
        current_vt = v_t(x[i-1], t)
        vt_drift = current_vt * dt                     # v_t单步增量（动态值）
        corr_drift = drift_correction(x[i-1], t) * dt  # 修正项单步增量
        diffusion = sigma_t(t) * np.random.normal(0, 1) * np.sqrt(dt)  # 扩散项单步增量
        # 2. 更新x_t
        x[i] = x[i-1] + vt_drift + corr_drift + diffusion
        # 3. 累计贡献（从x0开始）
        vt_contrib[i] = vt_contrib[i-1] + vt_drift
        corr_contrib[i] = corr_contrib[i-1] + corr_drift
        diff_contrib[i] = diff_contrib[i-1] + diffusion
        # 4. 保存单步增量
        vt_step[i] = vt_drift
        corr_step[i] = corr_drift
        diff_step[i] = diffusion
    
    # 保存当前采样的结果
    forward_sde_samples.append(x)
    forward_sde_vt_drift.append(vt_contrib)
    forward_sde_correction_drift.append(corr_contrib)
    forward_sde_diffusion.append(diff_contrib)
    forward_sde_vt_step.append(vt_step)
    forward_sde_correction_step.append(corr_step)
    forward_sde_diffusion_step.append(diff_step)

# ===================== 4. 逆向过程计算（ODE + SDE） =====================
## 4.1 逆向 ODE（确定性，使用修正后的v_t的相反数）
reverse_ode_x = np.zeros(num_steps)
reverse_ode_x[-1] = x1  # 从噪声x1开始
for i in range(num_steps-2, -1, -1):
    t = t_list[i]
    current_vt = v_t(reverse_ode_x[i+1], t)  # 正向v_t
    reverse_ode_x[i] = reverse_ode_x[i+1] - current_vt * dt  # 逆向为 -v_t

## 4.2 逆向 SDE（随机，多次采样）
# 累计贡献（子图4用）
reverse_sde_samples = []               # 完整轨迹
reverse_sde_vt_drift = []              # 逆向v_t项累计贡献
reverse_sde_correction_drift = []      # 修正项累计贡献
reverse_sde_diffusion = []             # 扩散项累计贡献
# 单步增量（子图6用）
reverse_sde_vt_step = []               # 逆向v_t项单步增量
reverse_sde_correction_step = []       # 修正项单步增量
reverse_sde_diffusion_step = []        # 扩散项单步增量

for _ in range(num_samples):
    x = np.zeros(num_steps)
    x[-1] = x1  # 从噪声x1开始
    # 累计贡献初始化
    vt_contrib = np.zeros(num_steps)
    corr_contrib = np.zeros(num_steps)
    diff_contrib = np.zeros(num_steps)
    # 单步增量初始化（最后1步无增量，设为0）
    vt_step = np.zeros(num_steps)
    corr_step = np.zeros(num_steps)
    diff_step = np.zeros(num_steps)
    
    for i in range(num_steps-2, -1, -1):
        t = t_list[i]
        # 1. 计算单步增量（逆向v_t为正向v_t的相反数）
        current_vt = v_t(x[i+1], t)
        vt_drift = -current_vt * dt                    # 逆向v_t单步增量（符号反转）
        corr_drift = drift_correction(x[i+1], t) * dt  # 修正项单步增量
        diffusion = sigma_t(t) * np.random.normal(0, 1) * np.sqrt(dt)  # 扩散项单步增量
        # 2. 更新x_t
        x[i] = x[i+1] + vt_drift + corr_drift + diffusion
        # 3. 累计贡献（从x1开始）
        vt_contrib[i] = vt_contrib[i+1] + vt_drift
        corr_contrib[i] = corr_contrib[i+1] + corr_drift
        diff_contrib[i] = diff_contrib[i+1] + diffusion
        # 4. 保存单步增量
        vt_step[i] = vt_drift
        corr_step[i] = corr_drift
        diff_step[i] = diffusion
    
    # 保存当前采样的结果
    reverse_sde_samples.append(x)
    reverse_sde_vt_drift.append(vt_contrib)
    reverse_sde_correction_drift.append(corr_contrib)
    reverse_sde_diffusion.append(diff_contrib)
    reverse_sde_vt_step.append(vt_step)
    reverse_sde_correction_step.append(corr_step)
    reverse_sde_diffusion_step.append(diff_step)

# ===================== 5. 可视化（3行2列，新增子图5/6） =====================
plt.rcParams['font.size'] = 9
fig, axes = plt.subplots(3, 2, figsize=(16, 18))
# 子图索引：ax1(0,0) 正向ODE+SDE | ax2(0,1) 正向SDE累计贡献
#          ax3(1,0) 逆向ODE+SDE | ax4(1,1) 逆向SDE累计贡献
#          ax5(2,0) 正向SDE每步增量 | ax6(2,1) 逆向SDE每步增量
ax1, ax3 = axes[0]
ax2, ax4 = axes[1]
ax5, ax6 = axes[2]

# -------------------- 子图1：正向过程 ODE + SDE --------------------
ax1.plot(t_list, forward_ode_x, 'r-', linewidth=2, label='Forward ODE (Deterministic)')
for i, sample in enumerate(forward_sde_samples):
    ax1.plot(t_list, sample, 'b--', alpha=0.5, label=f'Forward SDE Sample {i+1}' if i==0 else "")
ax1.set_xlabel('Time t')
ax1.set_ylabel('x_t')
ax1.set_title('Forward Process: ODE (Deterministic) vs SDE (Stochastic, Multiple Samples)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# -------------------- 子图2：正向SDE 累计贡献（v_t+修正+扩散） --------------------
for i in range(num_samples):
    # 完整SDE轨迹（黑色实线）
    ax2.plot(t_list, forward_sde_samples[i], 'k-', alpha=0.8, label=f'SDE Sample {i+1}' if i==0 else "")
    # v_t项累计贡献（红色虚线）
    ax2.plot(t_list, x0 + forward_sde_vt_drift[i], 'r--', alpha=0.6, label=f'v_t Drift (Cumulative) {i+1}' if i==0 else "")
    # 修正项累计贡献（蓝色点线）
    ax2.plot(t_list, x0 + forward_sde_correction_drift[i], 'b:', alpha=0.6, label=f'Correction Drift (Cumulative) {i+1}' if i==0 else "")
    # 扩散项累计贡献（绿色点划线）
    ax2.plot(t_list, x0 + forward_sde_diffusion[i], 'g-.', alpha=0.6, label=f'Diffusion (Cumulative) {i+1}' if i==0 else "")
ax2.set_xlabel('Time t')
ax2.set_ylabel('x_t / Cumulative Contribution')
ax2.set_title('Forward SDE: Cumulative Contributions (v_t + Correction + Diffusion)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# -------------------- 子图3：逆向过程 ODE + SDE --------------------
ax3.plot(t_list, reverse_ode_x, 'r-', linewidth=2, label='Reverse ODE (Deterministic)')
for i, sample in enumerate(reverse_sde_samples):
    ax3.plot(t_list, sample, 'b--', alpha=0.5, label=f'Reverse SDE Sample {i+1}' if i==0 else "")
ax3.set_xlabel('Time t')
ax3.set_ylabel('x_t')
ax3.set_title('Reverse Process: ODE (Deterministic) vs SDE (Stochastic, Multiple Samples)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# -------------------- 子图4：逆向SDE 累计贡献（-v_t+修正+扩散） --------------------
for i in range(num_samples):
    # 完整SDE轨迹（黑色实线）
    ax4.plot(t_list, reverse_sde_samples[i], 'k-', alpha=0.8, label=f'SDE Sample {i+1}' if i==0 else "")
    # 逆向v_t项累计贡献（红色虚线）
    ax4.plot(t_list, x1 + reverse_sde_vt_drift[i], 'r--', alpha=0.6, label=f'-v_t Drift (Cumulative) {i+1}' if i==0 else "")
    # 修正项累计贡献（蓝色点线）
    ax4.plot(t_list, x1 + reverse_sde_correction_drift[i], 'b:', alpha=0.6, label=f'Correction Drift (Cumulative) {i+1}' if i==0 else "")
    # 扩散项累计贡献（绿色点划线）
    ax4.plot(t_list, x1 + reverse_sde_diffusion[i], 'g-.', alpha=0.6, label=f'Diffusion (Cumulative) {i+1}' if i==0 else "")
ax4.set_xlabel('Time t')
ax4.set_ylabel('x_t / Cumulative Contribution')
ax4.set_title('Reverse SDE: Cumulative Contributions (-v_t + Correction + Diffusion)')
ax4.legend()
ax4.grid(True, alpha=0.3)

# -------------------- 子图5：正向SDE 每步增量（v_t+修正+扩散） --------------------
for i in range(num_samples):
    # v_t项单步增量（红色虚线）
    ax5.plot(step_indices, forward_sde_vt_step[i], 'r--', alpha=0.6, label=f'v_t Drift (Step) {i+1}' if i==0 else "")
    # 修正项单步增量（蓝色点线）
    ax5.plot(step_indices, forward_sde_correction_step[i], 'b:', alpha=0.6, label=f'Correction Drift (Step) {i+1}' if i==0 else "")
    # 扩散项单步增量（绿色点划线）
    ax5.plot(step_indices, forward_sde_diffusion_step[i], 'g-.', alpha=0.6, label=f'Diffusion (Step) {i+1}' if i==0 else "")
ax5.set_xlabel('Iteration Step Index')
ax5.set_ylabel('Stepwise Increment')
ax5.set_title('Forward SDE: Stepwise Increment of v_t Drift, Correction Drift and Diffusion')
ax5.legend()
ax5.grid(True, alpha=0.3)

# -------------------- 子图6：逆向SDE 每步增量（-v_t+修正+扩散） --------------------
for i in range(num_samples):
    # 逆向v_t项单步增量（红色虚线）
    ax6.plot(step_indices, reverse_sde_vt_step[i], 'r--', alpha=0.6, label=f'-v_t Drift (Step) {i+1}' if i==0 else "")
    # 修正项单步增量（蓝色点线）
    ax6.plot(step_indices, reverse_sde_correction_step[i], 'b:', alpha=0.6, label=f'Correction Drift (Step) {i+1}' if i==0 else "")
    # 扩散项单步增量（绿色点划线）
    ax6.plot(step_indices, reverse_sde_diffusion_step[i], 'g-.', alpha=0.6, label=f'Diffusion (Step) {i+1}' if i==0 else "")
ax6.set_xlabel('Iteration Step Index')
ax6.set_ylabel('Stepwise Increment')
ax6.set_title('Reverse SDE: Stepwise Increment of -v_t Drift, Correction Drift and Diffusion')
ax6.legend()
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()