import gymnasium as gym
import time

def run_lunar_lander():
    """月球着陆器环境可视化"""
    print("\n=== 月球着陆器 (LunarLander-v3) ===")
    print("控制方式：")
    print("  0: 不操作")
    print("  1: 左引擎")
    print("  2: 主引擎")
    print("  3: 右引擎")
    
    env = gym.make('LunarLander-v3', render_mode="human")
    observation, info = env.reset()
    
    for step in range(1000):
        env.render()
        action = env.action_space.sample()  # 随机动作
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"着陆{'成功' if reward > 0 else '失败'}，得分: {reward:.1f}")
            observation, info = env.reset()
        
        time.sleep(0.01)
    
    env.close()

def run_bipedal_walker():
    """双足步行机器人环境可视化"""
    print("\n=== 双足步行机器人 (BipedalWalker-v3) ===")
    print("目标：控制机器人行走尽可能远的距离")
    
    env = gym.make('BipedalWalker-v3', render_mode="human")
    observation, info = env.reset()
    
    for step in range(2000):
        env.render()
        action = env.action_space.sample()  # 随机动作
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"步行结束，步数: {step+1}")
            observation, info = env.reset()
        
        time.sleep(0.01)
    
    env.close()

def run_car_racing():
    """赛车环境可视化"""
    print("\n=== 赛车环境 (CarRacing-v2) ===")
    print("目标：控制赛车尽可能快地行驶")
    
    env = gym.make('CarRacing-v2', render_mode="human")
    observation, info = env.reset()
    
    for step in range(1000):
        env.render()
        action = env.action_space.sample()  # 随机动作
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            print(f"赛车结束，得分: {reward:.1f}")
            observation, info = env.reset()
        
        time.sleep(0.01)
    
    env.close()

if __name__ == "__main__":
    print("高级环境可视化演示")
    print("按窗口关闭按钮或等待自动结束...")
    
    # 运行月球着陆器
    run_lunar_lander()
    
    # 运行双足步行机器人
    run_bipedal_walker()
    
    # 运行赛车环境
    run_car_racing()