import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# ======= 参数设置 =======
# t=0 时的数据分布参数 (你的假设)
mu_a, mu_b = 2.0, -2.0
sigma_0 = 1.0  # 你要求的方差为1 (标准差也为1)

# t=1 时的噪声分布参数 (根据原图猜测大致为标准差较大的正态分布)
sigma_1 = 2.5  

# ======= 定义流匹配的边缘分布 CDF =======
def marginal_cdf(x, t):
    """计算时间 t 时的边缘高斯混合分布的累积概率密度 (CDF)"""
    # 独立耦合下，均值和方差随时间线性/平方插值
    mu_a_t = (1 - t) * mu_a
    mu_b_t = (1 - t) * mu_b
    sigma_t = np.sqrt((1 - t)**2 * sigma_0**2 + t**2 * sigma_1**2)
    
    # 假设数据是 1:1 的混合高斯
    cdf_a = norm.cdf(x, loc=mu_a_t, scale=sigma_t)
    cdf_b = norm.cdf(x, loc=mu_b_t, scale=sigma_t)
    return 0.5 * cdf_a + 0.5 * cdf_b

def find_x_for_quantile(q, t):
    """根据分位数 q 寻找对应的坐标 x (即逆 CDF)"""
    # 使用 brentq 求解 f(x) = cdf(x, t) - q = 0
    return brentq(lambda x: marginal_cdf(x, t) - q, -15, 15)

# ======= 生成轨迹 =======
t_vals = np.linspace(0, 1, 100)
# 生成 40 条代表不同分位数的流线
quantiles = np.linspace(0.01, 0.99, 40) 

trajectories = np.zeros((len(quantiles), len(t_vals)))

# 计算每条流线在每个时间点的位置
for i, q in enumerate(quantiles):
    for j, t in enumerate(t_vals):
        trajectories[i, j] = find_x_for_quantile(q, t)

# ======= 绘图 =======
plt.figure(figsize=(10, 10))

for i in range(len(quantiles)):
    plt.plot(t_vals, trajectories[i], 'b--', alpha=0.5)

# 画出中心的星星（代表分布的均值中心）
plt.scatter([0, 0], [-2, 2], color='black', marker='*', s=200, zorder=5, label='Mean of x0 clusters')

plt.xlim(-0.05, 1.05)
plt.ylim(-5.5, 5.5)
plt.grid(True, alpha=0.3)
plt.legend()
plt.title("Flow Matching Vector Field (x0 variance = 1)")
plt.xlabel("Time (t)")
plt.ylabel("State (x)")
plt.show()