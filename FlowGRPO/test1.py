import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# ======= 参数设置 =======
mu_a, mu_b = 2.0, -2.0
sigma_0 = 0.1  # 为了看清汇聚效果，我们把方差设小一点 (模拟接近原图的锐利度)
sigma_1 = 2.5  

# 模拟 Flow-GRPO 训练后的结果：赋予 +2.0 极大的权重，-2.0 极小的权重
weight_a = 0.95  # 去往 2.0 的概率 (高奖励)
weight_b = 0.05  # 去往 -2.0 的概率 (大惩罚)

# ======= 定义流匹配的边缘分布 CDF (带权重) =======
def marginal_cdf_rl(x, t):
    mu_a_t = (1 - t) * mu_a
    mu_b_t = (1 - t) * mu_b
    sigma_t = np.sqrt((1 - t)**2 * sigma_0**2 + t**2 * sigma_1**2)
    
    cdf_a = norm.cdf(x, loc=mu_a_t, scale=sigma_t)
    cdf_b = norm.cdf(x, loc=mu_b_t, scale=sigma_t)
    # 按 RL 优化后的权重混合
    return weight_a * cdf_a + weight_b * cdf_b

def find_x_for_quantile(q, t):
    # 使用 brentq 寻找等分位数路径
    return brentq(lambda x: marginal_cdf_rl(x, t) - q, -15, 15)

# ======= 生成轨迹 =======
t_vals = np.linspace(0, 1, 100)
# 生成 40 条等概率分位的流线
quantiles = np.linspace(0.01, 0.99, 40) 

trajectories = np.zeros((len(quantiles), len(t_vals)))

for i, q in enumerate(quantiles):
    for j, t in enumerate(t_vals):
        trajectories[i, j] = find_x_for_quantile(q, t)

# ======= 绘图 =======
plt.figure(figsize=(10, 10))

for i in range(len(quantiles)):
    plt.plot(t_vals, trajectories[i], 'b--', alpha=0.6, linewidth=1.5)

# 画出目标点
plt.scatter([0], [2], color='red', marker='*', s=300, zorder=5, label='High Reward (x0=2)')
plt.scatter([0], [-2], color='gray', marker='x', s=100, zorder=5, label='Penalized (x0=-2)')

# 辅助线：中轴线
plt.axhline(0, color='black', linestyle=':', alpha=0.3)

plt.xlim(-0.05, 1.05)
plt.ylim(-5.5, 5.5)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title("Flow Matching Vector Field after Flow-GRPO\n(High Reward at +2.0, Penalty at -2.0)")
plt.xlabel("Time (t)  [0=Data, 1=Noise]")
plt.ylabel("State (x)")
plt.show()