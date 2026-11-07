"""PathPlanner 參考軌跡：讀機器人專案匯出的 CSV，以參考時間 τ 取樣。

CSV 由機器人專案的 RlExportTest 用 PathPlannerLib 與同一份 RobotConfig 生成 (5 ms 取樣)，
放在 data/robot-export/trajectories/。不在 Python 重寫 PathPlanner 的軌跡生成器，避免跟實機不一致。

時間縮放：策略決定 dτ/dt = s，空間路徑不變，參考速度要乘 s (sample 的 scale 參數)。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .robot_software import wrap_rad

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "robot-export" / "trajectories"


@dataclass(frozen=True)
class ReferenceState:
    """批次參考狀態，每個欄位形狀 (N,)。速度為場地座標。"""

    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    omega: np.ndarray


class Trajectory:
    def __init__(self, name: str, t, x, y, heading, vx, vy, omega):
        if len(t) < 2 or np.any(np.diff(t) < 0):
            raise ValueError(f"{name}：時間必須遞增且至少兩個點")
        self.name = name
        self.t = np.asarray(t, dtype=np.float64)
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        # 朝向先展開成連續值再插值，避免 ±π 交界處插出反方向
        self.heading_unwrapped = np.unwrap(np.asarray(heading, dtype=np.float64))
        self.vx = np.asarray(vx, dtype=np.float64)
        self.vy = np.asarray(vy, dtype=np.float64)
        self.omega = np.asarray(omega, dtype=np.float64)

        seg = np.hypot(np.diff(self.x), np.diff(self.y))
        self.arc_length = np.concatenate([[0.0], np.cumsum(seg)])

        # 算橫向誤差用的折線：依弧長每 2 cm 重新取樣 (原始點在低速時很密、高速時較疏)
        n_pts = max(int(np.ceil(self.arc_length[-1] / self.CT_SPACING_M)) + 1, 2)
        s = np.linspace(0.0, self.arc_length[-1], n_pts)
        keep = np.concatenate([[True], np.diff(self.arc_length) > 0])
        self._ct_s = s
        self._ct_x = np.interp(s, self.arc_length[keep], self.x[keep])
        self._ct_y = np.interp(s, self.arc_length[keep], self.y[keep])

    CT_SPACING_M = 0.02
    CT_WINDOW = 15  # 局部搜尋前後各 15 段 = ±0.3 m；hint 每週期更新，車子一個週期最多走約 0.1 m

    @classmethod
    def load(cls, name: str, data_dir: Path = DATA_DIR) -> "Trajectory":
        data = np.genfromtxt(data_dir / f"{name}.csv", delimiter=",", names=True)
        return cls(name, data["t"], data["x"], data["y"], data["heading"], data["vx"], data["vy"], data["omega"])

    @staticmethod
    def available(data_dir: Path = DATA_DIR) -> list[str]:
        return sorted(p.stem for p in data_dir.glob("*.csv") if not p.stem.startswith("_"))

    @property
    def total_time(self) -> float:
        return float(self.t[-1])

    @property
    def length(self) -> float:
        return float(self.arc_length[-1])

    @property
    def start_pose(self) -> np.ndarray:
        return np.array([self.x[0], self.y[0], wrap_rad(self.heading_unwrapped[0])])

    def sample(self, tau, scale=1.0) -> ReferenceState:
        """取樣參考時間 τ (超出範圍會夾到端點)。scale = 時間縮放 s，速度乘上 s。"""
        tau = np.clip(np.asarray(tau, dtype=np.float64), 0.0, self.t[-1])
        scale = np.asarray(scale, dtype=np.float64)
        return ReferenceState(
            x=np.interp(tau, self.t, self.x),
            y=np.interp(tau, self.t, self.y),
            heading=wrap_rad(np.interp(tau, self.t, self.heading_unwrapped)),
            vx=np.interp(tau, self.t, self.vx) * scale,
            vy=np.interp(tau, self.t, self.vy) * scale,
            omega=np.interp(tau, self.t, self.omega) * scale,
        )

    def lateral_from_end_line(self, px, py):
        """到「最後一段延長線」的垂直距離。終點速度不為 0 的路線，車子會衝過終點，過終點後的偏移只算橫向。"""
        dx, dy = self.x[-1] - self.x[-2], self.y[-1] - self.y[-2]
        norm = np.hypot(dx, dy)
        return np.abs((np.asarray(px) - self.x[-1]) * dy - (np.asarray(py) - self.y[-1]) * dx) / norm

    def cross_track(self, px, py, hint=None):
        """點到路線折線的最短距離與該處的路線進度 (弧長，m)。輸入 (N,)，輸出 (距離, 進度, 段索引)。

        用來判定「偏線」(獎勵與終止條件)，跟參考時間無關：只看車子離路線多遠。
        hint = 上一次回傳的段索引 (N,)：只在附近 ±CT_WINDOW 段搜尋，快很多，也不會在路線
        繞回自己附近時跳到另一段。hint=None 時整條搜尋。
        """
        px = np.asarray(px, dtype=np.float64)[:, None]
        py = np.asarray(py, dtype=np.float64)[:, None]
        n_seg = len(self._ct_x) - 1
        if hint is None:
            idx = np.broadcast_to(np.arange(n_seg), (px.shape[0], n_seg))
        else:
            offsets = np.arange(-self.CT_WINDOW, self.CT_WINDOW + 1)
            idx = np.clip(np.asarray(hint)[:, None] + offsets, 0, n_seg - 1)
        ax, ay = self._ct_x[idx], self._ct_y[idx]
        dx, dy = self._ct_x[idx + 1] - ax, self._ct_y[idx + 1] - ay
        seg_len_sq = dx * dx + dy * dy
        safe = np.where(seg_len_sq > 0.0, seg_len_sq, 1.0)
        u = np.clip(((px - ax) * dx + (py - ay) * dy) / safe, 0.0, 1.0)
        u = np.where(seg_len_sq > 0.0, u, 0.0)
        dist = np.hypot(px - (ax + u * dx), py - (ay + u * dy))
        best = np.argmin(dist, axis=1)
        rows = np.arange(px.shape[0])
        seg = idx[rows, best]
        progress = self._ct_s[seg] + u[rows, best] * np.sqrt(seg_len_sq[rows, best])
        return dist[rows, best], progress, seg
