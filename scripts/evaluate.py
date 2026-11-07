"""評估訓練好的模型 vs PathPlanner 基準 (B1)：

    python scripts/evaluate.py runs/blue3-3-v1/best.pt --episodes 1024
    python scripts/evaluate.py runs/blue3-3-v1/best.pt --path Blue1-3      # 拿去跑別條路線 (泛化)

用跟訓練不同的種子；同一組種子跑策略與基準，比較才公平。結果另存 runs/<名稱>/report_<路線>.json。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_auto.sim.env import ACTION_DIM, OBS_DIM, SwerveAutoEnv, evaluate_policy  # noqa: E402
from rl_auto.train.ppo import ActorCritic, RunningNorm  # noqa: E402


def load_policy(checkpoint: Path):
    data = torch.load(checkpoint, weights_only=False)
    cfg = data["config"]
    model = ActorCritic(OBS_DIM, ACTION_DIM, cfg["hidden"], cfg["init_log_std"])
    model.load_state_dict(data["model"])
    model.eval()
    norm = RunningNorm(OBS_DIM)
    norm.mean, norm.var = data["norm_mean"], data["norm_var"]

    def policy(obs):
        with torch.no_grad():
            return model.dist(torch.as_tensor(norm(obs), dtype=torch.float32)).mean.numpy()

    return policy, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--path", default=None, help="預設 = 訓練的路線")
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20_000)
    parser.add_argument("--randomize", type=float, default=1.0)
    args = parser.parse_args()

    policy, cfg = load_policy(args.checkpoint)
    path = args.path or cfg["path"]
    n = min(args.episodes, 512)
    zero = lambda o: np.zeros((o.shape[0], ACTION_DIM))
    results = {}
    for label, pol in (("PathPlanner (B1)", zero), ("RL 策略", policy)):
        env = SwerveAutoEnv(path, n_envs=n, seed=args.seed, randomize_scale=args.randomize)
        results[label] = evaluate_policy(env, pol, args.episodes)

    b, r = results["PathPlanner (B1)"], results["RL 策略"]
    print(f"路線 {path}（訓練路線 {cfg['path']}），{args.episodes} 回合，隨機化 {args.randomize}")
    print(f"{'':16s}{'成功率':>8s}{'時間(s)':>10s}{'偏線p95(cm)':>13s}{'終點誤差(cm)':>13s}")
    for label, x in results.items():
        print(f"{label:16s}{x['success_rate']*100:7.1f}%{x['mean_time_success']:10.2f}"
              f"{x['max_cross_track_p95']*100:13.1f}{x['mean_end_error_success']*100:13.1f}")
    saved = (b["mean_time_success"] - r["mean_time_success"]) / b["mean_time_success"] * 100
    print(f"省下時間 {saved:.1f}%，成功率 {(r['success_rate'] - b['success_rate'])*100:+.1f} 點")
    out = args.checkpoint.parent / f"report_{path}.json"
    out.write_text(json.dumps({"path": path, "train_path": cfg["path"], "episodes": args.episodes,
                               "randomize": args.randomize, "results": results,
                               "time_saved_percent": saved}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已存 {out}")


if __name__ == "__main__":
    main()
