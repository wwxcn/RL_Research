import time
import gymnasium as gym
 
# 生成环境
env = gym.make('CartPole-v1', render_mode='human')
# 环境初始化
state = env.reset()
# 循环交互
while True:
    # 渲染画面
    #env.render()
    # 从动作空间随机获取一个动作
    action = env.action_space.sample()
    # agent与环境进行一步交互
    state, reward, terminated, truncated, info = env.step(action)
    print('state = {0}; reward = {1}'.format(state, reward))
    # 判断当前episode 是否完成
    if terminated:
        print('terminated')
        break
    time.sleep(0.1)
# 环境结束
env.close()