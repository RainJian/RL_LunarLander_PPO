# -*- coding: utf-8 -*-
"""
录制最佳模型的着陆演示 GIF（render_mode="rgb_array"）
- 默认优先使用 1.5M 冠军模型
- 多 seed 采样若干局，按「落点居中 + 高分」选取最优一局：
    1. 优先：reward >= MIN_REWARD 且落点 |x| <= MAX_OFFSET 的局中取最高分
    2. 兜底：取最高分一局
- 用 PIL 输出 GIF（隔帧抽样 + 自适应调色板，控制文件体积）
- 可传 --seed 指定固定局
"""
import argparse
import os

import gymnasium as gym
from PIL import Image
from stable_baselines3 import PPO

CANDIDATES = [
    "./models/ppo_lunar_lander_1p5m.zip",  # 100 局复验冠军（220.7 ± 77.2）
    "./models/best/best_model.zip",        # EvalCallback 最佳（204.3 ± 85.3）
    "./models/ppo_lunar_lander_500k.zip",
    "./models/ppo_lunar_lander_final.zip",
]
OUT_PATH = "./lunar_lander_demo.gif"
N_ATTEMPTS = 12
FRAME_STRIDE = 2          # 每 2 帧取 1 帧，GIF 更紧凑
MAX_FRAMES = 400          # 抽样后最多 400 帧
DURATION_MS = 40          # 每帧时长
MIN_REWARD = 200.0        # 展示局的最低得分要求
MAX_OFFSET = 0.35         # 落点与中心允许的最大偏移（米，旗帜在 ±2m）


def pick_model():
    for c in CANDIDATES:
        if os.path.exists(c):
            return c
    raise FileNotFoundError("没有找到可用于录制的模型。")


def run_episode(env, model, seed):
    """跑一局，返回 (frames, total_reward, final_x)。final_x 单位为米。"""
    obs, _ = env.reset(seed=seed)
    done = truncated = False
    frames, total_reward = [], 0.0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        frames.append(env.render())
    final_x = env.unwrapped.lander.position.x / 30.0  # Box2D 单位 -> 米
    return frames, total_reward, final_x


def save_gif(frames, reward, final_x):
    images = [Image.fromarray(f) for f in frames[::FRAME_STRIDE][:MAX_FRAMES]]
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
    print(f"本局得分 {reward:.1f}，落点偏移 x={final_x:+.2f} m（旗帜在 ±2.0 m）")


def main():
    parser = argparse.ArgumentParser(description="录制着陆演示 GIF")
    parser.add_argument("--seed", type=int, default=None, help="指定固定种子局")
    parser.add_argument("--model", default=None, help="模型路径（默认自动选择）")
    args = parser.parse_args()

    model_path = args.model or pick_model()
    print(f"使用模型: {model_path}")
    model = PPO.load(model_path, device="cpu")
    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    if args.seed is not None:
        frames, reward, final_x = run_episode(env, model, args.seed)
        save_gif(frames, reward, final_x)
        env.close()
        return

    # 自动选局：居中且高分优先
    best_centered, best_overall = None, None
    for seed in range(N_ATTEMPTS):
        frames, reward, final_x = run_episode(env, model, seed)
        print(f"seed={seed:>2}: 得分 {reward:7.1f}, 落点 x={final_x:+6.3f} m, {len(frames)} 帧")
        if best_overall is None or reward > best_overall[1]:
            best_overall = (seed, reward, frames, final_x)
        if reward >= MIN_REWARD and abs(final_x) <= MAX_OFFSET:
            if best_centered is None or reward > best_centered[1]:
                best_centered = (seed, reward, frames, final_x)

    seed, reward, frames, final_x = best_centered or best_overall
    print(f"选用 seed={seed}: 得分 {reward:.1f}, 落点偏移 {final_x:+.3f} m")
    save_gif(frames, reward, final_x)
    env.close()


if __name__ == "__main__":
    main()
