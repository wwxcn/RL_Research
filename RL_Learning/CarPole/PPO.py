import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
from collections import deque

class Actor(nn.Module):
    """
    策略网络（Actor）：根据当前状态输出动作概率分布
    """
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(64, 64)         # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(64, action_dim) # 隐藏层2到输出层
        self.relu = nn.ReLU()                # 激活函数
        self.softmax = nn.Softmax(dim=-1)    # 输出动作概率分布
    
    def forward(self, x):
        """前向传播：输入状态，输出动作概率"""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

class Critic(nn.Module):
    """
    价值网络（Critic）：评估当前状态的价值
    """
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)  # 输入层到隐藏层1
        self.fc2 = nn.Linear(64, 64)         # 隐藏层1到隐藏层2
        self.fc3 = nn.Linear(64, 1)          # 隐藏层2到输出层（状态价值）
        self.relu = nn.ReLU()                # 激活函数
    
    def forward(self, x):
        """前向传播：输入状态，输出状态价值"""
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class PPO:
    """
    近端策略优化（PPO）算法实现
    """
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, update_epochs=10):
        self.actor = Actor(state_dim, action_dim)  # 策略网络
        self.critic = Critic(state_dim)             # 价值网络
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)  # 策略网络优化器
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr) # 价值网络优化器
        self.gamma = gamma                          # 折扣因子
        self.gae_lambda = gae_lambda                # GAE的lambda参数
        self.clip_epsilon = clip_epsilon            # PPO裁剪系数
        self.update_epochs = update_epochs          # 每次更新迭代次数
        
        # 存储经验轨迹
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.next_states = []
        self.dones = []
    
    def select_action(self, state):
        """根据当前状态选择动作，并返回动作和对应的对数概率"""
        state = torch.tensor(state, dtype=torch.float32)
        probs = self.actor(state)       # 获取动作概率分布
        dist = Categorical(probs)       # 创建分类分布
        action = dist.sample()          # 采样动作
        log_prob = dist.log_prob(action)# 计算动作的对数概率
        return action.item(), log_prob.item()
    
    def store_transition(self, state, action, reward, log_prob, next_state, done):
        """存储一条经验轨迹"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def compute_gae_advantages(self, values):
        """
        计算GAE（广义优势估计）
        
        参数:
            values: 当前轨迹各状态的价值估计
            
        返回:
            advantages: GAE优势估计
            returns: 回报（用于训练价值网络）
        """
        advantages = []
        last_advantage = 0
        
        # 从后往前计算GAE优势
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                # 最后一个状态的下一状态价值为0（根据done标志）
                next_value = 0
            else:
                next_value = values[t + 1]
            
            # TD误差
            delta = self.rewards[t] + self.gamma * next_value * (1 - self.dones[t]) - values[t]
            # GAE优势计算
            last_advantage = delta + self.gamma * self.gae_lambda * (1 - self.dones[t]) * last_advantage
            advantages.insert(0, last_advantage)
        
        # 计算回报（用于训练价值网络）
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = advantages + values # ???? 应该是因为A_t ≈ R_t - V(s_t)
        
        # 标准化优势估计
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def update(self):
        """更新PPO策略网络和价值网络（使用GAE）"""
        # 转换为张量
        states = torch.tensor(np.array(self.states), dtype=torch.float32)
        actions = torch.tensor(self.actions, dtype=torch.int64)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float32)
        
        # 计算当前价值网络的价值估计
        with torch.no_grad():
            values = self.critic(states).squeeze()
        
        # 使用GAE计算优势和回报
        advantages, returns = self.compute_gae_advantages(values)
        
        # 多次迭代更新
        for _ in range(self.update_epochs):
            probs = self.actor(states)       # 当前策略的动作概率
            dist = Categorical(probs)        # 创建分类分布
            new_log_probs = dist.log_prob(actions) # 当前策略下动作的对数概率
            entropy = dist.entropy()         # 计算熵，用于鼓励探索，熵正则化策略
            
            values_pred = self.critic(states).squeeze() # 当前状态价值预测
            
            # PPO裁剪损失
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()
            
            # 价值网络损失
            critic_loss = nn.MSELoss()(values_pred, returns)
            
            # 更新策略网络
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 更新价值网络
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
        
        # 清空经验缓存
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.next_states = []
        self.dones = []

def main():
    """
    主函数：PPO算法训练CartPole-v1环境
    """
    # 创建CartPole环境
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]  # 状态维度
    action_dim = env.action_space.n             # 动作维度
    
    # 初始化PPO算法（使用GAE）
    ppo = PPO(state_dim, action_dim, gae_lambda=0.95)
    max_episodes = 1000  # 训练总轮数
    max_steps = 500      # 每轮最大步数
    
    # 用于记录训练过程
    reward_history = []       # 每轮奖励历史
    avg_reward_history = []   # 滑动平均奖励历史
    window_size = 100         # 滑动窗口大小
    reward_window = deque(maxlen=window_size)  # 滑动窗口队列
    
    # 开始训练
    for episode in range(max_episodes):
        state = env.reset()
        # 处理gymnasium返回的元组格式
        if isinstance(state, tuple):
            state = state[0]
        total_reward = 0  # 累计奖励
        
        for step in range(max_steps):
            # 选择动作
            action, log_prob = ppo.select_action(state)
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated  # 终止标志
            # 处理gymnasium返回的元组格式
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            total_reward += reward  # 累计奖励
            
            # 存储轨迹
            ppo.store_transition(state, action, reward, log_prob, next_state, done)
            
            # 更新状态
            state = next_state
            
            # 当回合结束或达到最大步数时更新策略
            if done or step == max_steps - 1:
                ppo.update()  # 用旧策略采样出完整轨迹后，更新PPO策略
                print(f"Episode {episode+1}, Total Reward: {total_reward}")
                # 记录奖励
                reward_history.append(total_reward)
                reward_window.append(total_reward)
                # 计算滑动平均奖励
                if len(reward_window) == window_size:
                    avg_reward = np.mean(reward_window)
                    avg_reward_history.append(avg_reward)
                break
    
    # 关闭环境
    env.close()
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 6))
    
    # 原始奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(reward_history, label='Total Reward', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward History')
    plt.legend()
    plt.grid(True)
    
    # 平均奖励曲线
    plt.subplot(1, 2, 2)
    plt.plot(avg_reward_history, label=f'Average Reward (window={window_size})', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Average Reward History')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ppo_training_curve.png')  # 保存图像
    plt.show()  # 显示图像
    
    # 推理示例：可视化渲染控制过程
    print("\n开始推理示例...")
    env = gym.make('CartPole-v1', render_mode='human')
    state = env.reset(seed=42)
    if isinstance(state, tuple):
        state = state[0]
    total_reward = 0
    
    # 可以自定义初始状态
    # 示例：设置初始状态为小车位置0.5，速度0，杆角度0.1，角速度0
    # state = np.array([0.0, 0, 0.1, 0], dtype=np.float32)
    
    for step in range(500):
        env.render()  # 渲染画面
        action, _ = ppo.select_action(state)
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