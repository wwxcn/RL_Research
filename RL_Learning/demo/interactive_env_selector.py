'''
### 主流环境分类：
经典控制环境 ：
1. CartPole-v1 - 倒立摆平衡问题
2. MountainCar-v0 - 山地车爬坡（离散动作）
3. MountainCarContinuous-v0 - 山地车爬坡（连续动作）
4. Acrobot-v1 - 双连杆摆问题
5. Pendulum-v1 - 单摆控制问题
其他热门环境 ：
1. LunarLander-v3 - 月球着陆器
2. BipedalWalker-v3 - 双足步行机器人
3. CarRacing-v3 - 赛车环境
'''

import gymnasium as gym
import time
import pygame

def run_env(env_id, max_steps=500):
    """运行指定环境并展示可视化"""
    print(f"\n=== 正在运行: {env_id} ===")
    print("提示：按 ESC 或 Q 键可退出并返回选择菜单")
    try:
        env = gym.make(env_id, render_mode="human")
        observation, info = env.reset()
        
        # 打印环境信息
        print(f"动作空间: {env.action_space}")
        print(f"观测空间: {env.observation_space}")
        
        running = True
        for step in range(max_steps):
            if not running:
                break
                
            env.render()
            action = env.action_space.sample()  # 随机动作
            observation, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"环境终止，步数: {step+1}")
                observation, info = env.reset()
            
            # 检测键盘事件
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                        print("\n用户请求退出，返回选择菜单...")
                        running = False
                        break
            
            time.sleep(0.01)
        
        env.close()
        print(f"环境运行结束（共执行 {step+1} 步）")
    except Exception as e:
        print(f"运行 {env_id} 失败: {e}")
        print("可能需要安装额外依赖，请运行: pip install gymnasium[all]")

def show_env_info(env_id):
    """显示环境详细信息"""
    try:
        env = gym.make(env_id)
        print(f"\n=== {env_id} 详细信息 ===")
        print(f"描述: {env.spec.description if hasattr(env.spec, 'description') else '无'}")
        print(f"动作空间: {env.action_space}")
        print(f"观测空间: {env.observation_space}")
        print(f"奖励范围: {env.reward_range}")
        print(f"最大步数: {env.spec.max_episode_steps if hasattr(env.spec, 'max_episode_steps') else '无'}")
        env.close()
    except Exception as e:
        print(f"获取 {env_id} 信息失败: {e}")

def main():
    """主函数，提供交互式环境选择"""
    # 环境分类
    env_categories = {
        "经典控制": [
            "CartPole-v1",
            "MountainCar-v0",
            "MountainCarContinuous-v0",
            "Acrobot-v1",
            "Pendulum-v1"
        ],
        "高级环境": [
            "LunarLander-v3",
            "BipedalWalker-v3",
            "CarRacing-v3"
        ]
    }
    
    while True:
        print("\n" + "="*50)
        print("Gymnasium 交互式环境选择器")
        print("="*50)
        
        # 显示菜单
        print("\n请选择操作:")
        print("1. 查看所有环境")
        print("2. 按类别浏览环境")
        print("3. 搜索特定环境")
        print("4. 运行环境可视化")
        print("5. 退出")
        
        choice = input("\n请输入选择 (1-5): ")
        
        if choice == "1":
            # 查看所有环境
            print("\n=== 所有可用环境 ===")
            for category, envs in env_categories.items():
                print(f"\n{category}:")
                for i, env_id in enumerate(envs, 1):
                    print(f"  {i}. {env_id}")
                    
        elif choice == "2":
            # 按类别浏览
            print("\n=== 环境类别 ===")
            for i, category in enumerate(env_categories.keys(), 1):
                print(f"  {i}. {category}")
            
            cat_choice = input("\n请输入类别编号: ")
            try:
                cat_index = int(cat_choice) - 1
                category = list(env_categories.keys())[cat_index]
                envs = env_categories[category]
                
                print(f"\n=== {category} 环境 ===")
                for i, env_id in enumerate(envs, 1):
                    print(f"  {i}. {env_id}")
                    
                # 显示选中环境的详细信息
                env_choice = input("\n查看环境详细信息 (输入编号，或回车跳过): ")
                if env_choice:
                    try:
                        env_index = int(env_choice) - 1
                        env_id = envs[env_index]
                        show_env_info(env_id)
                    except (ValueError, IndexError):
                        print("无效的环境编号")
                        
            except (ValueError, IndexError):
                print("无效的类别编号")
                
        elif choice == "3":
            # 搜索特定环境
            search_term = input("\n请输入环境名称或关键词: ")
            found = False
            
            print(f"\n=== 搜索结果 (包含 '{search_term}') ===")
            for category, envs in env_categories.items():
                for env_id in envs:
                    if search_term.lower() in env_id.lower():
                        print(f"  {env_id} ({category})")
                        found = True
            
            if not found:
                print(f"  未找到包含 '{search_term}' 的环境")
                
        elif choice == "4":
            # 运行环境可视化
            print("\n=== 选择要运行的环境 ===")
            all_envs = []
            for category, envs in env_categories.items():
                all_envs.extend(envs)
            
            for i, env_id in enumerate(all_envs, 1):
                print(f"  {i}. {env_id}")
            
            env_choice = input("\n请输入环境编号: ")
            try:
                env_index = int(env_choice) - 1
                env_id = all_envs[env_index]
                
                max_steps = input("请输入最大步数 (默认 500): ")
                max_steps = int(max_steps) if max_steps.isdigit() else 500
                
                run_env(env_id, max_steps)
                
            except (ValueError, IndexError):
                print("无效的环境编号")
                
        elif choice == "5":
            # 退出
            print("\n感谢使用 Gymnasium 环境选择器！")
            break
            
        else:
            print("\n无效的选择，请输入 1-5 之间的数字")
            
        # 等待用户确认
        input("\n按回车键继续...")

if __name__ == "__main__":
    main()