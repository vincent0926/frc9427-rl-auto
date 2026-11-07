"""開始訓練：python scripts/train.py --path Blue3-3 --steps 30000000

訓練過程寫到 runs/<名稱>/，用 scripts/dashboard.py 開監控網頁看。
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rl_auto.train.ppo import TrainConfig, Trainer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="Blue3-3")
    parser.add_argument("--steps", type=int, default=TrainConfig.total_steps)
    parser.add_argument("--envs", type=int, default=TrainConfig.n_envs)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()
    cfg = TrainConfig(path=args.path, total_steps=args.steps, n_envs=args.envs, seed=args.seed)
    Trainer(cfg, args.name).train()


if __name__ == "__main__":
    main()
