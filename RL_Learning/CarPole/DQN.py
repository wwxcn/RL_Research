import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque

class QNetwork(nn.Module):
    """
    Q 网络：输入状态，输出每个动作的 Q 值
    """
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)   # 输入层到隐藏层1
        self.fc2 = nn.Linear(64, 64)          # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(64, action_dim)  # 隐藏层2到输出层（每个动作的 Q 值）
        self.relu = nn.ReLU()                 # 激活函数

    def forward(self, x):
        """前向传播：输入状态，输出各动作的 Q 值"""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class ReplayBuffer:
    """
    经验回放缓存：存储 (s, a, r, s', done) 五元组，用于打乱相关性
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)  # 使用 deque，超出容量自动丢弃最旧数据

    def push(self, state, action, reward, next_state, done):
        """存入一条经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机采样一个批次的经验"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)  # (batch, 1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

class DQN:
    """
    Deep Q-Network（DQN）算法实现

    核心思想：
    1. 使用神经网络拟合 Q 值函数 Q(s, a)
    2. 经验回放（Experience Replay）：打乱样本相关性，稳定训练
    3. 目标网络（Target Network）：延迟更新目标 Q 网络，避免"移靶"问题
    4. ε-贪婪策略：平衡探索与利用
    """
    def __init__(self, state_dim, action_dim, lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 target_update_freq=10, buffer_capacity=10000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma                    # 折扣因子

        # 当前 Q 网络 + 目标 Q 网络
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_q_net = QNetwork(state_dim, action_dim)
        self.target_q_net.load_state_dict(self.q_net.state_dict())  # 初始权重相同
        self.target_q_net.eval()                                 # 目标网络不训练

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # ε-贪婪相关参数
        self.epsilon = epsilon                  # 初始探索率
        self.epsilon_min = epsilon_min          # 最小探索率
        self.epsilon_decay = epsilon_decay      # 探索率衰减

        # 训练相关参数
        self.target_update_freq = target_update_freq  # 目标网络更新频率（步）
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state):
        """ε-贪婪策略选择动作"""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)  # 随机探索
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # 增加 batch 维
            with torch.no_grad():
                q_values = self.q_net(state)
            return q_values.argmax(dim=1).item()  # 选择 Q 值最大的动作

    def store_transition(self, state, action, reward, next_state, done):
        """存储一条经验到回放缓存"""
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self, step):
        """
        使用经验回放训练 Q 网络，周期性更新目标网络

        参数:
            step: 当前总步数，用于判断是否更新目标网络
        """
        if len(self.replay_buffer) < self.batch_size:
            return  # 缓存不足时不训练

        # 从回放缓存采样一个批次
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 计算当前 Q 值：Q(s, a; θ)
        current_q_values = self.q_net(states).gather(1, actions)  # (batch, 1)

        # 计算目标 Q 值：r + γ * max_a' Q(s', a'; θ⁻)
        with torch.no_grad():
            next_q_values = self.target_q_net(next_states).max(dim=1, keepdim=True)[0]  # (batch, 1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # 计算损失并更新当前 Q 网络
        loss = self.criterion(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 周期性地将当前 Q 网络的权重同步到目标 Q 网络
        if step % self.target_update_freq == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        # ε 衰减
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main():
    """
    主函数：DQN 算法训练 CartPole-v1 环境
    """
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]  # 状态维度
    action_dim = env.action_space.n             # 动作维度

    dqn = DQN(state_dim, action_dim)
    max_episodes = 1000  # 训练总轮数
    max_steps = 500      # 每轮最大步数

    reward_history = []       # 每轮奖励历史
    avg_reward_history = []   # 滑动平均奖励历史
    window_size = 100         # 滑动窗口大小
    reward_window = deque(maxlen=window_size)

    total_step = 0  # 用于目标网络更新频率计数

    # 开始训练
    for episode in range(max_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        total_reward = 0

        for step in range(max_steps):
            action = dqn.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            total_reward += reward

            dqn.store_transition(state, action, reward, next_state, done)
            total_step += 1

            # DQN 每步都尝试更新（只要缓存足够）
            dqn.update(total_step)

            state = next_state

            if done:
                print(f"Episode {episode+1}, Total Reward: {total_reward}, Epsilon: {dqn.epsilon:.3f}")
                reward_history.append(total_reward)
                reward_window.append(total_reward)
                if len(reward_window) == window_size:
                    avg_reward = np.mean(reward_window)
                    avg_reward_history.append(avg_reward)
                break

    env.close()

    # 绘制训练曲线
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(reward_history, label='Total Reward', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Reward History')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(avg_reward_history, label=f'Average Reward (window={window_size})', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('DQN Average Reward History')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('dqn_training_curve.png')
    plt.show()

    # 推理示例（使用贪婪策略，ε 设置为 0）
    print("\n开始推理示例...")
    env = gym.make('CartPole-v1', render_mode='human')
    state = env.reset(seed=42)
    if isinstance(state, tuple):
        state = state[0]
    total_reward = 0
    dqn.epsilon = 0.0  # 推理时完全使用 Q 网络，不探索

    for step in range(500):
        env.render()
        action = dqn.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        if isinstance(next_state, tuple):
            next_state = next_state[0]
        total_reward += reward
        state = next_state

        if done:
            break

    print(f"推理示例总奖励: {total_reward}")
    env.close()

if __name__ == '__main__':
    main()
