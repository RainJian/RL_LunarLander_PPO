import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# 1. 创建环境
# render_mode=None 表示训练时不弹窗，适合在 WSL/服务器上跑
env_id = "LunarLander-v3"
env = gym.make(env_id, render_mode=None)

# 2. 定义模型保存和日志路径
log_dir = "./logs/"
model_dir = "./models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# 3. 初始化 PPO 智能体 (Agent)
# MlpPolicy: 使用多层感知机(神经网络)来处理输入状态(坐标、速度等)
# verbose=1: 打印训练进度
# tensorboard_log: 指定日志目录，用于可视化
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=log_dir,
    learning_rate=0.0003,  # 学习率，决定了模型更新的步幅
    n_steps=2048,          # 每次更新前收集的数据点数量
    batch_size=64,         # 每次参数更新使用的数据样本量
    gamma=0.99             # 折扣因子，0.99表示模型很看重未来的长期奖励
)

print("---------------------------------------")
print(f"开始训练 Agent: {env_id}")
print(f"日志将保存至: {log_dir}")
print("---------------------------------------")

# 4. 开始训练
# total_timesteps: 总共训练的步数。
# 对于 LunarLander，通常 10万步(100k) 能看到初步效果，50万步以上效果较好。
TOTAL_TIMESTEPS = 100000

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    tb_log_name="PPO_LunarLander"
)

# 5. 保存模型
model_path = os.path.join(model_dir, "ppo_lunar_lander_final")
model.save(model_path)
print(f"训练完成！模型已保存至: {model_path}.zip")

# 6. 简单的测试代码，看看训练后的奖励
print("正在测试模型性能...")
obs, _ = env.reset()
total_reward = 0
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = env.step(action)
    total_reward += reward
    if done or truncated:
        obs, _ = env.reset()
        break
print(f"测试结束，单次着陆获得奖励: {total_reward:.2f}")

env.close()