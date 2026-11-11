# -*- coding: utf-8 -*-
"""
从 TensorBoard 事件文件导出训练曲线图片（用于 README / 报告）
- 自动合并 logs/ 下所有 PPO_LunarLander_* 运行的数据（第一轮 + 续训轮）
- 额外读取 EvalCallback 的 evaluations.npz，绘制 eval/mean_reward ± std
输出到 Images/，覆盖旧图。
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOG_DIR = "./logs/"
EVAL_NPZ_GLOB = "./logs/eval/evaluations*.npz"   # 含各轮备份，按 timestep 合并
OUT_DIR = "./Images/"

SMOOTH = 0.92  # 平滑系数，模仿 TensorBoard 的 smoothed 曲线


def load_scalar(tag):
    """合并所有 run 中某个 tag 的数据，按 step 排序去重。"""
    points = {}
    for run_dir in sorted(glob.glob(os.path.join(LOG_DIR, "PPO_LunarLander_*"))):
        event_files = glob.glob(os.path.join(run_dir, "events.out.tfevents.*"))
        if not event_files:
            continue
        acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})
        acc.Reload()
        if tag not in acc.Tags().get("scalars", []):
            continue
        for e in acc.Scalars(tag):
            points[e.step] = e.value  # 后来的 run 覆盖同 step 旧值
    steps = sorted(points)
    return np.array(steps), np.array([points[s] for s in steps])


def smooth(values, weight=SMOOTH):
    out = np.empty_like(values, dtype=float)
    acc = values[0]
    for i, v in enumerate(values):
        acc = weight * acc + (1 - weight) * v
        out[i] = acc
    return out


def plot(tag, title, ylabel, out_name, ref_line=None, smooth_curve=True):
    steps, values = load_scalar(tag)
    if len(steps) == 0:
        print(f"[跳过] 未找到数据: {tag}")
        return
    fig, ax = plt.subplots(figsize=(8, 5), dpi=110)
    ax.plot(steps, values, color="#9bb7d4", alpha=0.55, linewidth=1, label="raw")
    if smooth_curve:
        ax.plot(steps, smooth(values), color="#1f4e9c", linewidth=2, label="smoothed")
    else:
        ax.plot(steps, values, color="#1f4e9c", linewidth=2)
    if ref_line is not None:
        ax.axhline(ref_line, color="#c0392b", linestyle="--", linewidth=1.2,
                   label=f"solved = {ref_line}")
    ax.axvline(100_352, color="#7f8c8d", linestyle=":", linewidth=1.2,
               label="continue from 100k")
    ax.set_title(title)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, out_name))
    plt.close(fig)
    print(f"[完成] {out_name}  ({len(steps)} 点, 末值 {values[-1]:.2f} @ {steps[-1]})")


def plot_eval():
    """EvalCallback 独立评估曲线：eval/mean_reward ± std —— 最有说服力的证据。
    合并各轮训练的 evaluations*.npz（含手动备份），按 timestep 对齐。"""
    points = {}  # timestep -> results array
    for path in sorted(glob.glob(EVAL_NPZ_GLOB)):
        data = np.load(path)
        for t, res in zip(data["timesteps"], data["results"]):
            points[int(t)] = np.asarray(res)
    if not points:
        print("[跳过] 未找到任何 evaluations npz")
        return
    timesteps = np.array(sorted(points))
    results = np.stack([points[t] for t in timesteps])
    means = results.mean(axis=1)
    stds = results.std(axis=1)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=110)
    ax.plot(timesteps, means, color="#c0392b", marker="o", markersize=4,
            linewidth=2, label="eval mean_reward (20 episodes)")
    ax.fill_between(timesteps, means - stds, means + stds,
                    color="#c0392b", alpha=0.18, label="±1 std")
    ax.axhline(200, color="#27ae60", linestyle="--", linewidth=1.4,
               label="solved = 200")
    ax.axvline(100_352, color="#7f8c8d", linestyle=":", linewidth=1.2,
               label="continue from 100k")
    ax.set_title("Independent Evaluation (EvalCallback, deterministic)")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Reward")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "eval_mean_reward.png"))
    plt.close(fig)
    print(f"[完成] eval_mean_reward.png  ({len(timesteps)} 评估点, "
          f"最终 {means[-1]:.1f} ± {stds[-1]:.1f}, 峰值 {means.max():.1f})")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plot("rollout/ep_rew_mean", "Episode Reward (rollout/ep_rew_mean)",
         "Mean Reward", "ep_rew_mean.png", ref_line=200)
    plot("rollout/ep_len_mean", "Episode Length (rollout/ep_len_mean)",
         "Mean Length", "ep_len_mean.png", smooth_curve=True)
    plot("train/value_loss", "Value Loss (train/value_loss)",
         "Loss", "value_loss.png", smooth_curve=False)
    plot("train/explained_variance", "Explained Variance (train/explained_variance)",
         "Explained Variance", "explained_variance.png", smooth_curve=False)
    plot("train/entropy_loss", "Entropy Loss (train/entropy_loss)",
         "Entropy Loss", "entropy_loss.png", smooth_curve=False)
    plot_eval()


if __name__ == "__main__":
    main()
