"""把訓練監控網頁發布到 GitHub Pages (docs/ 資料夾)：

    python scripts/publish_dashboard.py                 # 發布一次 (所有訓練的快照)
    python scripts/publish_dashboard.py --watch 5       # 訓練中每 5 分鐘自動推一次，GitHub 上就能看進度

只複製監控需要的小檔案 (status、metrics、基準、軌跡)，模型檔不放進 docs/。
網址：https://vincent0926.github.io/frc9427-rl-auto/
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
DOCS = ROOT / "docs"
FILES = ("status.json", "metrics.jsonl", "baseline.json", "best_eval.json", "trace.json", "baseline_trace.json")


def build() -> int:
    DOCS.mkdir(exist_ok=True)
    shutil.copy(ROOT / "dashboard" / "index.html", DOCS / "index.html")
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")  # 讓 GitHub Pages 直接提供檔案
    out_runs = DOCS / "runs"
    statuses = []
    if RUNS.exists():
        for run in sorted(RUNS.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            status = run / "status.json"
            if not run.is_dir() or not status.exists():
                continue
            target = out_runs / run.name
            target.mkdir(parents=True, exist_ok=True)
            for name in FILES:
                if (run / name).exists():
                    shutil.copy(run / name, target / name)
            statuses.append(json.loads(status.read_text(encoding="utf-8")))
    out_runs.mkdir(parents=True, exist_ok=True)
    index = {"published": time.time(), "runs": statuses}
    (out_runs / "index.json").write_text(json.dumps(index), encoding="utf-8")
    return len(statuses)


def git(*args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")


def publish() -> None:
    count = build()
    git("add", "docs")
    if git("diff", "--cached", "--quiet", "--", "docs").returncode == 0:
        print("沒有變更，不用推送")
        return
    stamp = time.strftime("%Y-%m-%d %H:%M")
    commit = git("commit", "-m", f"Update training dashboard snapshot ({stamp})", "--", "docs")
    if commit.returncode != 0:
        print(commit.stdout, commit.stderr)
        return
    push = git("push")
    print(f"已發布 {count} 個訓練 ({stamp})" if push.returncode == 0 else f"推送失敗：{push.stderr}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch", type=float, default=0, help="每幾分鐘自動發布一次 (0 = 只發布一次)")
    args = parser.parse_args()
    publish()
    while args.watch > 0:
        time.sleep(args.watch * 60)
        publish()


if __name__ == "__main__":
    sys.exit(main())
