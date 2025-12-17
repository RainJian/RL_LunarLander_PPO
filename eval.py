import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np

# 1. 设置路径
model_path = "./models/ppo_lunar_lander_final.zip"

# 2. 创建环境
env = gym.make("LunarLander-v3", render_mode=None)

# 3. 加载模型
print(f"正在加载模型: {model_path} ...")
try:
    model = PPO.load(model_path)
    print("模型加载成功！")
except FileNotFoundError:
    print(f"错误：找不到模型文件。请检查 {model_path} 是否存在。")
    exit()

# 4. 运行评估 (跑 5 局取平均分)
n_episodes = 5
print(f"\n开始评估模型 (共 {n_episodes} 局)...")
print("-" * 30)

rewards = []
for i in range(n_episodes):
    obs, _ = env.reset()
    done = False
    truncated = False
    episode_reward = 0

    while not (done or truncated):
        # 关键：deterministic=True 使用最优策略
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward

    rewards.append(episode_reward)
    print(f"第 {i + 1} 局得分: {episode_reward:.2f}")

env.close()

# 5. 输出最终结果
mean_reward = np.mean(rewards)
std_reward = np.std(rewards)

print("-" * 30)
print(f"评估完成！")
print(f"平均得分: {mean_reward:.2f} +/- {std_reward:.2f}")
print("-" * 30)

if mean_reward > 190:
    print("结果评级: 优秀")
else:
    print("结果评级: 需改进")