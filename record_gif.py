# -*- coding: utf-8 -*-
"""
录制最佳模型的着陆演示 GIF（render_mode="rgb_array"）
- 优先使用 EvalCallback 保存的 best model
- 跑若干局，选取 得分>=200 且长度适中 的一局；找不到则选得分最高的一局
- 用 PIL 输出 GIF（隔帧抽样 + 自适应调色板，控制文件体积）
"""
import os

import gymnasium as gym
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

CANDIDATES = [
    "./models/ppo_lunar_lander_1p5m.zip",  # 100 局复验冠军（220.7 ± 77.2）
    "./models/best/best_model.zip",        # EvalCallback 最佳（204.3 ± 85.3）
    "./models/ppo_lunar_lander_500k.zip",
    "./models/ppo_lunar_lander_final.zip",
]
OUT_PATH = "./lunar_lander_demo.gif"
N_ATTEMPTS = 10
FRAME_STRIDE = 2          # 每 2 帧取 1 帧，GIF 更紧凑
MAX_FRAMES = 400          # 抽样后最多 400 帧
DURATION_MS = 40          # 每帧时长


def pick_model():
    for c in CANDIDATES:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("没有找到可用于录制的模型。")


def run_episode(env, model):
    obs, _ = env.reset()
    done = truncated = False
    frames, total_reward = [], 0.0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        frames.append(env.render())
    return frames, total_reward


def main():
    model_path = pick_model()
    print(f"使用模型: {model_path}")
    model = PPO.load(model_path, device="cpu")
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    best = None  # (reward, frames)
    for i in range(N_ATTEMPTS):
        frames, reward = run_episode(env, model)
        print(f"第 {i + 1} 局: 得分 {reward:.2f}, {len(frames)} 帧")
        if best is None or reward > best[0]:
            best = (reward, frames)
        if reward >= 200 and len(frames) // FRAME_STRIDE <= MAX_FRAMES:
            best = (reward, frames)
            break
    env.close()

    reward, frames = best
    print(f"选用局: 得分 {reward:.2f}, 原始 {len(frames)} 帧")

    images = []
    for f in frames[::FRAME_STRIDE][:MAX_FRAMES]:
        img = Image.fromarray(f)
        images.append(img)

    images[0].save(
        OUT_PATH,
        save_all=True,
        append_images=images[1:],
        duration=DURATION_MS * FRAME_STRIDE,
        loop=0,
        optimize=True,
    )
    size_mb = os.path.getsize(OUT_PATH) / 1024 / 1024
    print(f"GIF 已保存: {OUT_PATH} ({len(images)} 帧, {size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
