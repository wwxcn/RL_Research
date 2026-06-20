# CartPole 强化学习算法合集

本目录包含三种经典强化学习算法在 **CartPole-v1** 环境的 PyTorch 实现，以及一份横向对比文档，旨在清晰呈现 **值函数方法（DQN）** 与 **策略梯度方法（PPO / GRPO）** 的架构差异。

## 目录结构

```
CarPole/
├── PPO.py       # 近端策略优化（含 GAE）
├── GRPO.py      # 组相对策略优化
├── DQN.py       # 深度 Q 网络
└── README.md    # 本文档
```

## 运行方式

```bash
python PPO.py    # 训练并推理，输出 ppo_training_curve.png
python GRPO.py   # 训练并推理，输出 grpo_training_curve.png
python DQN.py    # 训练并推理，输出 dqn_training_curve.png
```

依赖：`gymnasium`、`numpy`、`torch`、`matplotlib`、`random`、`collections`。

---

## 1. DQN（Deep Q-Network）—— 值函数方法

**文件**：`DQN.py`

### 整体架构

```
┌────────────────────────────────────────────────────────────────┐
│                       DQN 交互循环                              │
│                                                                │
│   env ──s_t──► ε-greedy 选择 a_t  ──► env.step() ──► r_t, s'   │
│                         │                                      │
│                         └──► store(s_t, a_t, r_t, s', done)    │
│                                │                               │
│                                ▼                               │
│                    ReplayBuffer (deque)                        │
│                    ├── (s1, a1, r1, s1', done)                 │
│                    ├── (s2, a2, r2, s2', done)                 │
│                    └── ...                                     │
│                                │                               │
│                     random.sample(batch_size=64)               │
│                                │                               │
│                                ▼                               │
│                ┌─────────────────────────────┐                 │
│                │   Bellman 方程训练          │                 │
│                │   L = MSE(Q_target, Q_current) │               │
│                └─────────────────────────────┘                 │
│                    │                    │                       │
│                    ▼                    ▼                       │
│              q_net（当前网络）  target_q_net（目标网络）        │
│             每步反向传播更新    每 target_update_freq 步硬同步  │
└────────────────────────────────────────────────────────────────┘
```

### 关键实现思考

1. **两套 Q 网络 + 硬同步**（核心）
   - `q_net` 用于计算当前 Q 值并做反向传播
   - `target_q_net` 用于计算 Bellman 目标中的 `max Q(s', a')`
   - 每隔 `target_update_freq`（默认 10）步调用 `load_state_dict()` 做一次硬同步
   - 不使用目标网络会出现"打移动靶"现象，训练极易发散

2. **经验回放（Replay Buffer）**
   - 使用 `collections.deque(maxlen=capacity)` 实现
   - 训练时 `random.sample()` 随机抽 64 条样本
   - 打破 MDP 序列样本的时间相关性，是 DQN 稳定训练的两大基石之一

3. **ε-贪婪策略 + 探索率衰减**
   - 训练初期 `epsilon=1.0`（全探索），每步乘以 `epsilon_decay=0.995`
   - 衰减到 `epsilon_min=0.01` 后停止，保留少量探索
   - 推理时手动设置 `epsilon=0.0`，完全依赖 Q 网络的贪婪选择

4. **Bellman 方程的数值细节**
   - `gather(1, actions)`：从 `(batch, action_dim)` 中取出实际执行动作对应的 Q 值
   - 目标值：`r + γ · max Q_target(s') · (1 - done)`，`done=1` 时终止态的后续价值为 0
   - `target_q_net` 计算时必须 `with torch.no_grad()`，否则会把目标网络的梯度也纳入

---

## 2. PPO（Proximal Policy Optimization）—— 策略梯度方法

**文件**：`PPO.py`

### 整体架构

```
┌────────────────────────────────────────────────────────────────┐
│                       PPO 交互循环                              │
│                                                                │
│   env ──s_t──► Actor(π_θ) 采样 a_t ~ π(a|s)                    │
│                │         ↓ 返回 log_prob(a_t)                  │
│                ├──► env.step() ──► r_t, s'                     │
│                └──► 存储 (s_t, a_t, r_t, log_prob_t, s', done) │
│                                                                │
│   ── 轨迹结束 ───────────────────────────────────────────►     │
│                                                                │
│   计算 GAE 优势 A_t 与回报 R_t（需要 Critic V(s)）              │
│   L_actor = -E[ min(ratio · A, clip(ratio, 1-ε, 1+ε) · A) ]    │
│              - 0.01 · E[ Entropy(π_θ) ]                        │
│   L_critic = MSE( V_φ(s), R_t )                                │
│                                                                │
│   对同一批轨迹进行 update_epochs（默认 10）次迭代更新            │
│   ├── Actor 用 Adam(lr=3e-4) 更新策略权重                      │
│   └── Critic 用 Adam(lr=3e-4) 更新价值权重                     │
│                                                                │
│   清空轨迹缓存，进入下一轮交互                                  │
└────────────────────────────────────────────────────────────────┘
```

### 关键实现思考

1. **GAE（广义优势估计）** 替代简单 `R_t - V(s_t)`
   - `delta_t = r_t + γ · V(s_{t+1}) - V(s_t)`（TD 误差）
   - `A_t = sum_{l=0}^∞ (γ·λ)^l · delta_{t+l}`（指数加权）
   - `gae_lambda=0.95`：平衡低偏差（λ=0 对应 1 步 TD）与低方差（λ=1 对应 Monte Carlo）
   - 计算回报时用 `R_t = A_t + V(s_t)`，训练 Critic
   - 对所有优势做**全局**零均值标准化

2. **裁剪损失（核心创新）**
   - `ratio = exp(log_prob_new - log_prob_old)`，衡量新旧策略差异
   - 取 `min(ratio·A, clip(ratio, 1-ε, 1+ε)·A)`：保守地取较"小"的那个
   - `clip_epsilon=0.2` 是业界标准值，保证每次更新不会让策略突变
   - 解决了 Vanilla Policy Gradient "一步更新过猛导致训练崩溃"的问题

3. **熵正则化（鼓励探索）**
   - 在 Actor 损失中减去 `0.01 · E[Entropy(π)]`
   - 熵越大代表动作概率越均匀，避免策略过早陷入确定性策略

4. **多 epoch 复用同一批轨迹**
   - 对收集到的一条完整轨迹重复 `update_epochs=10` 次更新
   - 这是 PPO 数据效率高的原因：每条样本用 10 次而不是 1 次
   - 配合裁剪损失，保证即便多 epoch 也不会让策略偏离太远

5. **Actor-Critic 架构的价值**
   - Critic 估计状态价值 `V(s)`，作为"基线"（Baseline）
   - 优势 `A = Q(s, a) - V(s)` 的期望不变，但方差大幅降低
   - 注意：Critic 的目标值 `R_t` 也是从同一条轨迹计算而来，无需额外交互

---

## 3. GRPO（Group Relative Policy Optimization）—— 策略梯度方法的轻量变体

**文件**：`GRPO.py`

### 整体架构

```
┌────────────────────────────────────────────────────────────────┐
│                       GRPO 交互循环                             │
│                                                                │
│   ┌ Trajectory 1 ┐ ┌ Trajectory 2 ┐ ... ┌ Trajectory N ┐       │
│   │  (s, a, r)   │ │  (s, a, r)   │     │  (s, a, r)   │       │
│   │  (s, a, r)   │ │  (s, a, r)   │     │  (s, a, r)   │       │
│   │  ...         │ │  ...         │     │  ...         │       │
│   └── R_1 ──────┘ └── R_2 ──────┘     └── R_N ──────┘         │
│                          │                                      │
│                          ▼                                      │
│              "组"（Group）—— 共 group_size=8 条轨迹             │
│                          │                                      │
│                          ▼                                      │
│         1. 对每条轨迹计算累计回报 R_i = sum(γ^t · r_t)           │
│         2. 组内标准化：A_i = (R_i - mean(R)) / (std(R) + ε)    │
│         3. A_i 作为该轨迹内每一步的相对优势                      │
│                          │                                      │
│                          ▼                                      │
│          L = -E[ min(ratio·A, clip(ratio, 1-ε, 1+ε)·A) ]        │
│            - 0.01 · E[ Entropy ]                                │
│         （裁剪损失 + 熵正则化，与 PPO 形式相同）                  │
│                          │                                      │
│                          ▼                                      │
│                 仅更新 Actor 网络（无 Critic）                   │
└────────────────────────────────────────────────────────────────┘
```

### 关键实现思考

1. **去掉 Critic 的理由与代价**
   - GRPO 用**组内相对排名**代替 Critic 的价值估计
   - 优点：网络数量减半、更新速度更快、超参数更少
   - 缺点：优势估计的方差更大（整条轨迹共享同一个优势值），对长任务或稀疏奖励场景训练稳定性不如 PPO+GAE

2. **"组"是 GRPO 的核心概念**
   - `group_size=8`：每次更新必须先收集 8 条完整轨迹
   - 组内做零均值标准化：回报高于组均值的轨迹被"鼓励"（A > 0），低于组均值的轨迹被"惩罚"（A < 0）
   - 这种相对比较天然地充当了基线（Baseline）的角色

3. **轨迹级优势 vs 步骤级优势**
   - PPO+GAE：**每一步**都有自己的优势估计 `A_t`
   - GRPO：**同一条轨迹内的所有步骤**共享同一个优势值 `A_i`
   - 这意味着 GRPO 的信用分配比较粗糙：一条轨迹中"好动作"和"坏动作"被相同程度地增强/减弱

4. **裁剪损失完全沿用 PPO**
   - `ratio = exp(log_prob_new - log_prob_old)` 的形式一模一样
   - `clip_epsilon=0.2`、`update_epochs=10`、熵系数 `0.01` 全部沿用
   - 这意味着 GRPO 可视为 PPO 的"优势估计简化版"

5. **适用场景**
   - **适合**：短回合任务（如 CartPole）、奖励信号密集、对训练速度有要求
   - **慎选**：长回合任务、稀疏奖励、动作空间巨大的连续控制

---

## 4. 三种算法横向对比总结

| 维度 | DQN | PPO | GRPO |
|------|-----|-----|------|
| **算法家族** | 值函数方法 | 策略梯度（Actor-Critic） | 策略梯度（纯 Actor） |
| **网络** | Q 网络 + 目标 Q 网络 | Actor + Critic | 仅 Actor |
| **核心损失** | MSE（Bellman 方程） | PPO 裁剪损失 + Critic MSE | PPO 裁剪损失 |
| **优势估计** | 通过 Q 值隐式体现 | GAE 每步优势（全局标准化） | 轨迹累计回报（组内标准化） |
| **数据复用** | 经验回放（~每步都更新） | 多 epoch + 同批轨迹复用 | 组内多 epoch 复用 |
| **探索机制** | ε-贪婪衰减 | 熵正则化 + 策略随机性 | 熵正则化 + 策略随机性 |
| **训练稳定性** | 中等（依赖目标网络和回放） | 高（裁剪 + GAE 是 SOTA 组合） | 中（无 Critic，方差较大） |
| **每步更新** | ✅ 每步都更新 | ❌ 等整条轨迹结束 | ❌ 等一个"组"的轨迹结束 |
| **网络数量** | 2 套（同结构，硬同步） | 2 套（Actor / Critic，独立更新） | 1 套 |
| **适用动作空间** | 离散 | 离散 + 连续 | 离散 + 连续 |
| **CartPole 表现** | 收敛稳定，500 不难达成 | 收敛最稳最快，效果最好 | 稍慢，最终性能接近 PPO |

### 推荐阅读顺序

**DQN → PPO → GRPO**

- DQN：理解值函数、Bellman 方程、经验回放与目标网络的作用
- PPO：理解策略梯度、Actor-Critic 分工、GAE 与裁剪损失
- GRPO：理解如何通过"组内比较"替代 Critic 的价值估计，体会"简化版本"的取舍

### 扩展方向（可自行实践）

- **DQN 的改进**：Double DQN、Dueling DQN、PER（优先经验回放）、Noisy DQN
- **PPO 的改进**：连续动作空间（Gaussian 分布）、循环策略（LSTM/GRU）、KL 惩罚版 PPO
- **GRPO 的扩展**：加入 GAE（用 Critic 算每步优势后再组内标准化）、更大的 group_size
- **超参数调优**：把三个文件的学习率、batch size、group size、clip ε 做成 sweep，对比学习曲线
