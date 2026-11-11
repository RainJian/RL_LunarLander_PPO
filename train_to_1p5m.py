# -*- coding: utf-8 -*-
"""
PPO 微调脚本（第四轮：1M -> 1.5M 步，学习率退火）
触发原因：1M 时 eval/mean_reward 峰值 204.7 但震荡（189/172），
最佳模型 100 局复验 178.5，未达"稳定 >=200"验收线。
处置（计划文档风险预案：调参实验）：学习率由 2e-4 退火至 1e-4，
用更小更新步幅抑制后期策略震荡，再加训 500k 步。
"""
import os
import time

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

ENV_ID = "LunarLander-v3"
BASE_MODEL_PATH = "./models/ppo_lunar_lander_1m.zip"   # 第三轮 1M 最终模型
LOG_DIR = "./logs/"
MODEL_DIR = "./models/"
BEST_MODEL_DIR = "./models/best/"
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_lunar_lander_1p5m")

CONTINUE_TIMESTEPS = 500_000   # 再加训 500k，总步数约 1.52M
N_ENVS = 8
LEARNING_RATE = 1e-4           # 退火：抑制后期震荡
EVAL_FREQ = 10_000
N_EVAL_EPISODES = 20


def build_train_env():
    try:
        env = make_vec_env(ENV_ID, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)
        print(f"[环境] SubprocVecEnv 向量化成功: n_envs={N_ENVS}")
        return env
    except Exception as e:
        print(f"[环境] 向量化失败（{type(e).__name__}: {e}），退回单环境训练")
        return gym.make(ENV_ID, render_mode=None)


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    if not os.path.exists(BASE_MODEL_PATH):
        raise FileNotFoundError(f"找不到第三轮模型: {BASE_MODEL_PATH}")

    env = build_train_env()
    eval_env = gym.make(ENV_ID, render_mode=None)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=os.path.join(LOG_DIR, "eval"),
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=os.path.join(MODEL_DIR, "checkpoints"),
        name_prefix="ppo_lunar_lander",
    )

    print("---------------------------------------")
    print(f"加载第三轮模型: {BASE_MODEL_PATH}")
    model = PPO.load(
        BASE_MODEL_PATH,
        env=env,
        device="cpu",
        tensorboard_log=LOG_DIR,
        custom_objects={"learning_rate": LEARNING_RATE},
        print_system_info=False,
    )
    print(f"已训练步数: {model.num_timesteps}")
    print(f"微调参数: +{CONTINUE_TIMESTEPS} 步 | lr 退火至 {LEARNING_RATE}")
    print("---------------------------------------")

    t0 = time.time()
    model.learn(
        total_timesteps=CONTINUE_TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name="PPO_LunarLander",
        callback=[eval_callback, checkpoint_callback],
    )
    elapsed = time.time() - t0

    model.save(FINAL_MODEL_PATH)
    print("---------------------------------------")
    print(f"微调完成！本次耗时 {elapsed/60:.1f} 分钟，总步数 {model.num_timesteps}")
    print(f"最终模型: {FINAL_MODEL_PATH}.zip")
    print("---------------------------------------")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
