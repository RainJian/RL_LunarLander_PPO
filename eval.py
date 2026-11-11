# -*- coding: utf-8 -*-
"""
模型评估脚本（第二轮更新版）
- 默认评估 EvalCallback 保存的最佳模型，找不到时自动退回 500k 最终模型 / 100k 初代模型
- 输出 均值 ± 标准差、最高/最低分、达标局数（单局 >=200 记为一次成功着陆）
用法:
    python eval.py                       # 评估 best model，20 局
    python eval.py --episodes 100        # 评估 100 局（对应官方 solved 判定口径）
    python eval.py --model ./models/ppo_lunar_lander_500k.zip
"""
import argparse
import os

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

CANDIDATES = [
    "./models/ppo_lunar_lander_1p5m.zip", # 1.5M 最终模型（100 局复验冠军 220.7 ± 77.2）
    "./models/best/best_model.zip",       # EvalCallback 保存的最佳模型
    "./models/ppo_lunar_lander_500k.zip", # 第二轮 500k 最终模型
    "./models/ppo_lunar_lander_final.zip",# 第一轮 100k 模型
]


def resolve_model(path: str) -> str:
    if os.path.exists(path):
        return path
    for c in CANDIDATES:
        if os.path.exists(c):
            print(f"未找到 {path}，自动退回: {c}")
            return c
    raise FileNotFoundError("没有找到任何可评估的模型文件。")


def main():
    parser = argparse.ArgumentParser(description="评估 PPO LunarLander 模型")
    parser.add_argument("--model", default=CANDIDATES[0], help="模型路径")
    parser.add_argument("--episodes", type=int, default=20, help="评估局数")
    args = parser.parse_args()

    model_path = resolve_model(args.model)

    env = gym.make("LunarLander-v3", render_mode=None)
    print(f"正在加载模型: {model_path} ...")
    model = PPO.load(model_path, device="cpu")
    print("模型加载成功！")

    print(f"\n开始评估 (共 {args.episodes} 局, deterministic=True)...")
    print("-" * 44)

    rewards, lengths = [], []
    for i in range(args.episodes):
        obs, _ = env.reset()
        done = truncated = False
        episode_reward, steps = 0.0, 0

        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

        rewards.append(episode_reward)
        lengths.append(steps)
        print(f"第 {i + 1:>3} 局 | 得分: {episode_reward:8.2f} | 步数: {steps}")

    env.close()

    mean_reward = float(np.mean(rewards))
    std_reward = float(np.std(rewards))
    n_solved = sum(1 for r in rewards if r >= 200)

    print("-" * 44)
    print("评估完成！")
    print(f"模型: {model_path}")
    print(f"平均得分: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"最高/最低: {max(rewards):.2f} / {min(rewards):.2f}")
    print(f"平均回合长度: {np.mean(lengths):.0f} 步")
    print(f"单局 >=200 分: {n_solved}/{args.episodes} 局")
    print("-" * 44)

    if mean_reward >= 200:
        print("结果评级: 优秀（达到官方 solved 标准）")
    else:
        print("结果评级: 需改进")


if __name__ == "__main__":
    main()
