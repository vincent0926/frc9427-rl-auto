"""感測模型與視覺融合 (設計文件 §3、§6；模擬器實作順序第 5 步)。

- VisionFusion：Swerve.addVisionMeasurement + PoseHistory 的批次移植 (延遲補償、首筆對齊、增益、跳動門檻)
- VisionCamera：Limelight 的抽象模型，不模擬標籤幾何：隨機頻率拍攝真實位置，加雜訊、延遲、掉幀
- Gyro：陀螺儀朝向 = 真實朝向 + 隨機遊走偏移
- CommandDelay：模組命令晚幾個週期才生效 (CAN、控制迴圈時序)

所有參數都可以是純量或形狀 (N,) 的陣列 (每個環境不同，給領域隨機化用)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .robot_software import Odometry, wrap_rad


@dataclass
class SensorParams:
    """預設值 = 設計文件 §6 隨機化範圍的中間值。關掉某項：vision_enabled=False、gyro_drift=0、command_delay_cycles=0。"""

    vision_enabled: bool = True
    vision_rate_hz: float | np.ndarray = 12.0  # Limelight 有效畫面頻率 (看得到標籤時)
    vision_latency_s: float | np.ndarray = 0.05  # 拍攝到 roboRIO 收到
    vision_sigma_m: float | np.ndarray = 0.04  # MegaTag1 位置雜訊 (每軸)
    vision_dropout: float | np.ndarray = 0.2  # 掉幀 / 看不到標籤的機率
    gyro_drift_rad_per_sqrt_s: float | np.ndarray = 0.002  # 隨機遊走強度
    command_delay_cycles: int | np.ndarray = 1  # 0–3


# Constants.Swerve 的視覺融合常數
VISION_BLEND_GAIN = 0.1
VISION_MAX_JUMP_M = 2.0
VISION_MAX_YAW_RATE_RAD = np.radians(60.0)


def _per_env(value, n, dtype=np.float64):
    return np.broadcast_to(np.asarray(value, dtype=dtype), (n,)).copy()


class PoseHistory:
    """PoseHistory.java 的批次版本：每個環境最近 capacity 筆 (時間, x, y)，線性內插取樣。"""

    def __init__(self, n_envs: int, capacity: int = 100):
        self.n = n_envs
        self.capacity = capacity
        self.times = np.full((n_envs, capacity), -np.inf)
        self.xs = np.zeros((n_envs, capacity))
        self.ys = np.zeros((n_envs, capacity))
        self.count = np.zeros(n_envs, dtype=int)
        self.head = 0  # 所有環境同步加入，共用寫入位置

    def clear(self, mask=None):
        mask = np.ones(self.n, dtype=bool) if mask is None else mask
        self.times[mask] = -np.inf
        self.count[mask] = 0

    def add(self, time, x, y):
        self.times[:, self.head] = time
        self.xs[:, self.head] = x
        self.ys[:, self.head] = y
        self.head = (self.head + 1) % self.capacity
        self.count = np.minimum(self.count + 1, self.capacity)

    def shift(self, dx, dy, mask):
        """視覺修正後，整段紀錄一起平移 (同 Java shift)，下一筆視覺才不會重複修正。"""
        self.xs[mask] += dx[mask, None]
        self.ys[mask] += dy[mask, None]

    def sample(self, t):
        """回傳 (x, y, 狀態)：狀態 0 = 範圍內、-1 = 比最舊還早、+1 = 比最新還新或沒有紀錄。"""
        # 環形緩衝區所有環境同步寫入，從 head 開始繞一圈就是由舊到新 (被清掉的是 -inf，排在最前面)
        order = (self.head + np.arange(self.capacity)) % self.capacity
        times = self.times[:, order]
        xs = self.xs[:, order]
        ys = self.ys[:, order]
        valid = np.isfinite(times)
        oldest = np.where(valid.any(axis=1), np.min(np.where(valid, times, np.inf), axis=1), np.inf)
        newest = np.max(times, axis=1)
        status = np.where(self.count == 0, 1, np.where(t < oldest, -1, np.where(t > newest, 1, 0)))

        # 向量化內插：排序後 (無效的 -inf 在最前面)，找 t 左右兩筆
        k = np.sum(times <= t[:, None], axis=1)  # 有幾筆 ≤ t
        newer = np.clip(k, 0, self.capacity - 1)[:, None]
        older = np.clip(k - 1, 0, self.capacity - 1)[:, None]
        t0 = np.take_along_axis(times, older, axis=1)[:, 0]
        t1 = np.take_along_axis(times, newer, axis=1)[:, 0]
        with np.errstate(invalid="ignore", divide="ignore"):  # 空的紀錄是 -inf，結果會被下面蓋掉
            span = t1 - t0
            frac = np.where(np.isfinite(span) & (span > 0.0), (t - t0) / np.where(span > 0.0, span, 1.0), 1.0)
        frac = np.clip(np.nan_to_num(frac, nan=1.0), 0.0, 1.0)
        x0 = np.take_along_axis(xs, older, axis=1)[:, 0]
        x1 = np.take_along_axis(xs, newer, axis=1)[:, 0]
        y0 = np.take_along_axis(ys, older, axis=1)[:, 0]
        y1 = np.take_along_axis(ys, newer, axis=1)[:, 0]
        return x0 + (x1 - x0) * frac, y0 + (y1 - y0) * frac, status


class VisionFusion:
    """Swerve.addVisionMeasurement 的批次版本，直接修改 Odometry 的位姿。"""

    def __init__(self, n_envs: int):
        self.n = n_envs
        self.history = PoseHistory(n_envs)
        self.has_fix = np.zeros(n_envs, dtype=bool)

    def reset(self, mask=None):
        mask = np.ones(self.n, dtype=bool) if mask is None else mask
        self.history.clear(mask)
        self.has_fix[mask] = False

    def record(self, time, odometry: Odometry):
        self.history.add(time, odometry.pose[:, 0], odometry.pose[:, 1])

    def add_measurement(self, odometry: Odometry, vx, vy, capture_time, mask):
        """mask：這個週期有收到 (且通過品質/轉速門檻) 的環境。回傳實際套用的 mask。"""
        bad = np.isnan(vx) | np.isnan(vy) | ((np.abs(vx) < 1e-4) & (np.abs(vy) < 1e-4))
        mask = mask & ~bad
        ref_x, ref_y, status = self.history.sample(capture_time)
        mask = mask & (status != -1)
        newer = status == 1
        ref_x = np.where(newer, odometry.pose[:, 0], ref_x)
        ref_y = np.where(newer, odometry.pose[:, 1], ref_y)
        dx = vx - ref_x
        dy = vy - ref_y
        mask = mask & (dx * dx + dy * dy <= VISION_MAX_JUMP_M ** 2)

        gain = np.where(self.has_fix, VISION_BLEND_GAIN, 1.0)
        cx = np.where(mask, dx * gain, 0.0)
        cy = np.where(mask, dy * gain, 0.0)
        odometry.pose[:, 0] += cx
        odometry.pose[:, 1] += cy
        self.history.shift(cx, cy, mask)
        self.has_fix |= mask
        odometry._refit(mask)
        return mask


class VisionCamera:
    """抽象的 Limelight：依頻率拍攝「真實位置 + 雜訊」，延遲後送到 roboRIO；一次只有一張在途中。"""

    def __init__(self, n_envs: int, params: SensorParams, rng: np.random.Generator):
        self.n = n_envs
        self.rng = rng
        self.period = 1.0 / _per_env(params.vision_rate_hz, n_envs)
        self.latency = _per_env(params.vision_latency_s, n_envs)
        self.sigma = _per_env(params.vision_sigma_m, n_envs)
        self.dropout = _per_env(params.vision_dropout, n_envs)
        self.next_capture = np.zeros(n_envs)
        self.pending = np.zeros(n_envs, dtype=bool)
        self.pending_xy = np.zeros((n_envs, 2))
        self.capture_time = np.zeros(n_envs)

    def reset(self, mask=None):
        """環境時間歸零時呼叫：第一張畫面的時間隨機落在一個週期內。"""
        mask = np.ones(self.n, dtype=bool) if mask is None else mask
        self.next_capture[mask] = self.rng.uniform(0.0, self.period)[mask]
        self.pending[mask] = False

    def update(self, time, true_pose):
        """拍攝新畫面；回傳這個週期送達的 (x, y, 拍攝時間, mask)。"""
        capture = (time >= self.next_capture) & ~self.pending
        keep = capture & (self.rng.random(self.n) >= self.dropout)
        noise = self.rng.normal(size=(self.n, 2)) * self.sigma[:, None]
        self.pending_xy[keep] = true_pose[keep, :2] + noise[keep]
        self.capture_time[keep] = time[keep]
        self.pending |= keep
        self.next_capture = np.where(capture, self.next_capture + self.period, self.next_capture)

        arrived = self.pending & (time >= self.capture_time + self.latency)
        self.pending &= ~arrived
        return self.pending_xy[:, 0].copy(), self.pending_xy[:, 1].copy(), self.capture_time.copy(), arrived


class Gyro:
    """陀螺儀朝向 = 真實朝向 + 隨機遊走偏移 (開機時歸零)。"""

    def __init__(self, n_envs: int, params: SensorParams, rng: np.random.Generator):
        self.rng = rng
        self.strength = _per_env(params.gyro_drift_rad_per_sqrt_s, n_envs)
        self.offset = np.zeros(n_envs)

    def reset(self, mask=None):
        if mask is None:
            self.offset[:] = 0.0
        else:
            self.offset[mask] = 0.0

    def read(self, true_heading, dt):
        self.offset += self.rng.normal(size=self.offset.shape) * self.strength * np.sqrt(dt)
        return wrap_rad(true_heading + self.offset)


class CommandDelay:
    """模組命令延遲 d 個週期 (每個環境不同)。d = 0 表示這個週期就生效。"""

    def __init__(self, n_envs: int, params: SensorParams, max_delay: int = 3):
        self.n = n_envs
        self.delay = np.clip(_per_env(params.command_delay_cycles, n_envs, dtype=int), 0, max_delay)
        self.size = max_delay + 1
        self.angle = np.zeros((self.size, n_envs, 4))
        self.speed = np.zeros((self.size, n_envs, 4))
        self.head = 0

    def reset(self, current_angle, mask=None):
        mask = np.ones(self.n, dtype=bool) if mask is None else mask
        self.angle[:, mask] = current_angle[mask][None]
        self.speed[:, mask] = 0.0

    def push(self, angle, speed):
        self.head = (self.head + 1) % self.size
        self.angle[self.head] = angle
        self.speed[self.head] = speed
        idx = (self.head - self.delay) % self.size
        rows = np.arange(self.n)
        return self.angle[idx, rows], self.speed[idx, rows]
