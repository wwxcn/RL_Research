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

class GRPO:
    """
    Group Relative Policy Optimization（GRPO）算法实现

    GRPO 的核心思想：
    - 使用一组（group）轨迹作为样本
    - 对组内每条轨迹的累计奖励进行零均值标准化，作为相对优势
    - 相比 PPO 的 GAE 优势估计，GRPO 更加简洁，无需 Critic 价值网络
    - 保持 PPO 的裁剪损失和熵正则化，保证训练稳定性
    """
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2, update_epochs=10):
        self.actor = Actor(state_dim, action_dim)                 # 策略网络
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)  # 策略网络优化器
        self.gamma = gamma                                         # 折扣因子
        self.clip_epsilon = clip_epsilon                           # PPO裁剪系数
        self.update_epochs = update_epochs                         # 每次更新迭代次数

        # 存储多条轨迹（一组）
        self.group_states = []      # 组内各轨迹的状态序列
        self.group_actions = []     # 组内各轨迹的动作序列
        self.group_rewards = []     # 组内各轨迹的奖励序列
        self.group_log_probs = []   # 组内各轨迹的对数概率序列

    def select_action(self, state):
        """根据当前状态选择动作，并返回动作和对应的对数概率"""
        state = torch.tensor(state, dtype=torch.float32)
        probs = self.actor(state)       # 获取动作概率分布
        dist = Categorical(probs)       # 创建分类分布
        action = dist.sample()          # 采样动作
        log_prob = dist.log_prob(action)# 计算动作的对数概率
        return action.item(), log_prob.item()

    def start_episode(self):
        """开始一条新轨迹：初始化当前轨迹的缓存"""
        self.cur_states = []
        self.cur_actions = []
        self.cur_rewards = []
        self.cur_log_probs = []

    def store_transition(self, state, action, reward, log_prob):
        """在当前轨迹中存储一条经验"""
        self.cur_states.append(state)
        self.cur_actions.append(action)
        self.cur_rewards.append(reward)
        self.cur_log_probs.append(log_prob)

    def finish_episode(self):
        """完成一条轨迹：将当前轨迹加入组中"""
        self.group_states.append(self.cur_states)
        self.group_actions.append(self.cur_actions)
        self.group_rewards.append(self.cur_rewards)
        self.group_log_probs.append(self.cur_log_probs)

    def compute_returns(self, rewards):
        """
        计算单条轨迹的折扣回报

        参数:
            rewards: 轨迹中每一步的奖励列表

        返回:
            returns: 每一步对应的折扣回报
        """
        returns = []
        running_return = 0
        for r in reversed(rewards):
            running_return = r + self.gamma * running_return
            returns.insert(0, running_return)
        return returns

    def compute_group_advantages(self):
        """
        计算组内相对优势（GRPO 的核心）

        步骤:
            1. 对组内每条轨迹计算累计回报（或累计奖励）
            2. 对累计回报在组内做零均值标准化，得到每条轨迹的相对优势
            3. 将每条轨迹的相对优势广播到其内部每一步

        返回:
            all_states: 所有轨迹拼接后的状态张量
            all_actions: 所有轨迹拼接后的动作张量
            all_old_log_probs: 所有轨迹拼接后的旧对数概率张量
            all_advantages: 所有轨迹拼接后的相对优势张量
        """
        # 计算每条轨迹的累计回报
        trajectory_returns = []
        trajectory_returns_list = []
        for rewards in self.group_rewards:
            step_returns = self.compute_returns(rewards)
            trajectory_returns_list.append(step_returns)
            trajectory_returns.append(step_returns[0])  # 用轨迹首步回报作为整段轨迹的"得分"

        # 对组内累计回报进行零均值标准化，得到组内相对优势
        returns_tensor = torch.tensor(trajectory_returns, dtype=torch.float32)
        group_mean = returns_tensor.mean()
        group_std = returns_tensor.std() + 1e-8
        group_advantages = (returns_tensor - group_mean) / group_std

        # 收集所有轨迹的数据，将每步优势设置为对应轨迹的组内相对优势
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_advantages = []

        for i in range(len(self.group_states)):
            traj_len = len(self.group_states[i])
            # 该轨迹的每一步都共享同一个相对优势
            adv = group_advantages[i].item()
            for t in range(traj_len):
                all_states.append(self.group_states[i][t])
                all_actions.append(self.group_actions[i][t])
                all_old_log_probs.append(self.group_log_probs[i][t])
                all_advantages.append(adv)

        all_states = torch.tensor(np.array(all_states), dtype=torch.float32)
        all_actions = torch.tensor(all_actions, dtype=torch.int64)
        all_old_log_probs = torch.tensor(all_old_log_probs, dtype=torch.float32)
        all_advantages = torch.tensor(all_advantages, dtype=torch.float32)

        return all_states, all_actions, all_old_log_probs, all_advantages

    def update(self):
        """使用 GRPO 更新策略网络"""
        states, actions, old_log_probs, advantages = self.compute_group_advantages()

        # 多次迭代更新（保持 PPO 的多 epoch 风格）
        for _ in range(self.update_epochs):
            probs = self.actor(states)              # 当前策略的动作概率
            dist = Categorical(probs)               # 创建分类分布
            new_log_probs = dist.log_prob(actions)  # 当前策略下动作的对数概率
            entropy = dist.entropy()                # 计算熵，用于鼓励探索

            # GRPO 裁剪损失（与 PPO 的形式一致，只是优势来源不同）
            ratio = torch.exp(new_log_probs - old_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy.mean()

            # 更新策略网络
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

        # 清空组经验缓存
        self.group_states = []
        self.group_actions = []
        self.group_rewards = []
        self.group_log_probs = []

def main():
    """
    主函数：GRPO 算法训练 CartPole-v1 环境
    """
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]  # 状态维度
    action_dim = env.action_space.n             # 动作维度

    grpo = GRPO(state_dim, action_dim)
    max_episodes = 1000   # 训练总轮数
    max_steps = 500       # 每轮最大步数
    group_size = 8        # 每次更新所需的轨迹数量（组大小）

    reward_history = []       # 每轮奖励历史
    avg_reward_history = []   # 滑动平均奖励历史
    window_size = 100         # 滑动窗口大小
    reward_window = deque(maxlen=window_size)

    # 开始训练
    for episode in range(max_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        total_reward = 0
        grpo.start_episode()

        for step in range(max_steps):
            action, log_prob = grpo.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            total_reward += reward

            grpo.store_transition(state, action, reward, log_prob)

            state = next_state

            if done or step == max_steps - 1:
                grpo.finish_episode()
                print(f"Episode {episode+1}, Total Reward: {total_reward}")
                reward_history.append(total_reward)
                reward_window.append(total_reward)
                if len(reward_window) == window_size:
                    avg_reward = np.mean(reward_window)
                    avg_reward_history.append(avg_reward)
                break

        # 当收集到足够多的轨迹时，进行一次 GRPO 更新
        if len(grpo.group_states) >= group_size or episode == max_episodes - 1:
            grpo.update()

    env.close()

    # 绘制训练曲线
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(reward_history, label='Total Reward', alpha=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('GRPO Reward History')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(avg_reward_history, label=f'Average Reward (window={window_size})', color='red')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('GRPO Average Reward History')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('grpo_training_curve.png')
    plt.show()

    # 推理示例
    print("\n开始推理示例...")
    env = gym.make('CartPole-v1', render_mode='human')
    state = env.reset(seed=42)
    if isinstance(state, tuple):
        state = state[0]
    total_reward = 0

    for step in range(500):
        env.render()
        action, _ = grpo.select_action(state)
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
