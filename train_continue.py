# -*- coding: utf-8 -*-
"""
PPO 续训脚本（第二轮：100k -> 500k 步）
依据《PPO 项目续训至收敛计划》编写：
  1. PPO.load 加载第一轮模型，set_env 后继续 learn（reset_num_timesteps=False）
  2. 续训 400k 步，使总步数达到 500k（SB3 官方对 LunarLander 的建议量）
  3. 加入 EvalCallback：每 10k 步独立评估 20 局，自动保存最佳模型
  4. learning_rate 由 3e-4 降至 2e-4，提高后期稳定性
  5. 8 进程向量化环境加速采样（失败时退回 DummyVecEnv / 单环境）
  6. MlpPolicy 为小网络，瓶颈在 CPU 物理仿真，故显式使用 CPU 训练
"""
import os
import time

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# ---------------- 配置 ----------------
ENV_ID = "LunarLander-v3"
BASE_MODEL_PATH = "./models/ppo_lunar_lander_final.zip"   # 第一轮 100k 步模型
LOG_DIR = "./logs/"
MODEL_DIR = "./models/"
BEST_MODEL_DIR = "./models/best/"
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_lunar_lander_500k")

CONTINUE_TIMESTEPS = 400_000   # 续训 400k，累计约 500k
N_ENVS = 8                     # 向量化环境数量
LEARNING_RATE = 2e-4           # 较第一轮 3e-4 略降，求稳
EVAL_FREQ = 10_000             # 每 10k 步评估一次
N_EVAL_EPISODES = 20           # 每次评估 20 局


def build_train_env():
    """优先 8 进程并行采样；向量化报错则退回单环境（风险预案）。"""
    try:
        env = make_vec_env(ENV_ID, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
        print(f"[环境] SubprocVecEnv 向量化成功: n_envs={N_ENVS}")
        return env
    except Exception as e:
        print(f"[环境] 向量化失败（{type(e).__name__}: {e}），退回单环境训练")
        return gym.make(ENV_ID, render_mode=None)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(f"找不到第一轮模型: {BASE_MODEL_PATH}")

    # 1. 训练环境 + 独立评估环境
    env = build_train_env()
    eval_env = gym.make(ENV_ID, render_mode=None)

    # 2. EvalCallback：计划的核心改进
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=os.path.join(LOG_DIR, "eval"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )
    # 保险：每 100k 步存一个检查点
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=os.path.join(MODEL_DIR, "checkpoints"),
        name_prefix="ppo_lunar_lander",
    )

    # 3. 加载第一轮模型，降学习率续训；MlpPolicy 小网络用 CPU 更快
    print("---------------------------------------")
    print(f"加载基线模型: {BASE_MODEL_PATH}")
    model = PPO.load(
        BASE_MODEL_PATH,
        env=env,
        device="cpu",
        tensorboard_log=LOG_DIR,
        custom_objects={"learning_rate": LEARNING_RATE},
        print_system_info=True,
    )
    print(f"已训练步数: {model.num_timesteps}")
    print(f"续训参数: +{CONTINUE_TIMESTEPS} 步 | lr={LEARNING_RATE} | "
          f"n_steps={model.n_steps} batch_size={model.batch_size} gamma={model.gamma}")
    print("---------------------------------------")

    # 4. 续训：reset_num_timesteps=False 使步数/日志从 100k 继续累计
    t0 = time.time()
    model.learn(
        total_timesteps=CONTINUE_TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name="PPO_LunarLander",
        callback=[eval_callback, checkpoint_callback],
    )
    elapsed = time.time() - t0

    # 5. 保存最终模型
    model.save(FINAL_MODEL_PATH)
    print("---------------------------------------")
    print(f"续训完成！本次耗时 {elapsed/60:.1f} 分钟，总步数 {model.num_timesteps}")
    print(f"最终模型: {FINAL_MODEL_PATH}.zip")
    print(f"最佳模型: {BEST_MODEL_DIR}best_model.zip（由 EvalCallback 保存）")
    print("---------------------------------------")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
