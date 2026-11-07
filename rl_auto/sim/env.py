"""向量化訓練環境 (設計文件 §8；模擬器實作順序第 6 步)。

一次跑 N 個環境，介面類似 Gymnasium VectorEnv (自動重置)：
    env = SwerveAutoEnv("Blue3-3", n_envs=1024, seed=0)
    obs = env.reset()                                   # (N, OBS_DIM)
    obs, reward, terminated, truncated, info = env.step(action)   # action: (N, 4)，範圍 [-1, 1]

動作 (殘差策略，規格書 §4)：
    a[0] → 時間縮放 s = 1 + 0.5·a[0] ∈ [0.5, 1.5]：沿路線推進多快，空間路徑不變
    a[1:3] → 機器人座標速度修正 Δvx, Δvy (最多 ±0.5 m/s × residual_scale)
    a[3] → 角速度修正 Δω (最多 ±1 rad/s × residual_scale)
    動作全 0 = 原本的 PathPlanner (基準 B1)。residual_scale = 0 時只有時間縮放 (課程學習第一階段)。

觀測只用機器人拿得到的資訊 (估計位姿、量到的模組速度、參考軌跡、電池電壓)；
獎勵與終止用真實位姿 (模擬器才知道)，策略才會學會避免「里程計看起來沒事、其實已經偏了」。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chassis import RealisticPlant
from .params import DEFAULT_PARAMS, RobotParams
from .randomization import Randomizer, apply_physics, apply_sensors
from .robot_software import chassis_from_modules, field_to_robot, robot_to_field, wrap_rad
from .rollout import PathFollower
from .sensors import SensorParams
from .trajectory import Trajectory

LOOKAHEAD_S = (0.25, 0.5, 1.0)
OBS_DIM = 3 + 3 + 3 + 2 * len(LOOKAHEAD_S) + len(LOOKAHEAD_S) + 1 + 1 + 4
ACTION_DIM = 4


@dataclass
class RewardConfig:
    progress: float = 10.0  # 走完整條路線 (真實進度) 共得幾分
    time_per_baseline: float = 5.0  # 每經過「一倍 PathPlanner 時間」扣幾分 → 越快完成越好
    cross_track_per_s: float = 2.0  # 偏線懲罰：每秒扣 w·(ct / 0.1 m)²
    action_change: float = 0.05  # 動作變化懲罰 (避免抖動)
    success_bonus: float = 2.0
    end_error_per_m: float = 20.0  # 終點誤差 (射擊位置) 每公尺扣幾分
    failure: float = 10.0  # 偏線超過門檻終止時扣分


class SwerveAutoEnv:
    def __init__(self, path: str = "Blue3-3", n_envs: int = 256, seed: int = 0,
                 randomize_scale: float = 1.0, use_sensors: bool = True, residual_scale: float = 1.0,
                 cross_track_limit_m: float = 0.3, finish_tolerance_m: float = 0.05,
                 max_time_factor: float = 1.5, reward: RewardConfig | None = None,
                 params: RobotParams = DEFAULT_PARAMS):
        self.traj = Trajectory.load(path) if isinstance(path, str) else path
        self.n = n_envs
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.randomizer = Randomizer(self.rng, randomize_scale, params)
        self.residual_scale = residual_scale
        self.ct_limit = cross_track_limit_m
        self.finish_tol = finish_tolerance_m
        self.max_time = self.traj.total_time * max_time_factor + 1.0
        self.reward_cfg = reward or RewardConfig()
        self.sensor_defaults = SensorParams() if use_sensors else None

        self.plant = RealisticPlant(n_envs, params)
        self.follower = PathFollower(self.traj, n_envs, params, plant=self.plant,
                                     sensors=self.sensor_defaults, seed=seed + 1)
        self.end = np.array([self.traj.x[-1], self.traj.y[-1]])

        self.prev_action = np.zeros((n_envs, ACTION_DIM))
        self.ct_hint = np.zeros(n_envs, dtype=int)
        self.progress = np.zeros(n_envs)
        self.ep_return = np.zeros(n_envs)
        self.ep_max_ct = np.zeros(n_envs)

    # ---- 重置 -------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self._reset(np.ones(self.n, dtype=bool))
        return self._observe()

    def _reset(self, mask):
        k = int(mask.sum())
        if k == 0:
            return
        apply_physics(self.plant, self.randomizer.sample_physics(self.n), mask)
        if self.sensor_defaults is not None:
            apply_sensors(self.follower, self.randomizer.sample_sensors(self.n, self.sensor_defaults), mask)
        offset = self.randomizer.sample_start_offset(self.n)
        self.follower.reset(mask, start_offset=offset)
        pose = self.plant.pose
        _, prog, hint = self.traj.cross_track(pose[mask, 0], pose[mask, 1])
        self.ct_hint[mask] = hint
        self.progress[mask] = prog
        self.prev_action[mask] = 0.0
        self.ep_return[mask] = 0.0
        self.ep_max_ct[mask] = 0.0

    # ---- 一步 ---------------------------------------------------------------
    def step(self, action):
        cfg = self.reward_cfg
        dt = self.params.cycle_time_s
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)

        time_scale = 1.0 + 0.5 * a[:, 0]
        est_heading = self.follower.odometry.pose[:, 2]
        dvx_r = 0.5 * self.residual_scale * a[:, 1]
        dvy_r = 0.5 * self.residual_scale * a[:, 2]
        dvx, dvy = robot_to_field(dvx_r, dvy_r, est_heading)
        residual = np.column_stack([dvx, dvy, 1.0 * self.residual_scale * a[:, 3]])

        self.follower.step(time_scale, residual)

        pose = self.plant.pose
        ct, progress, self.ct_hint = self.traj.cross_track(pose[:, 0], pose[:, 1], self.ct_hint)
        past_end = progress >= self.traj.length - 1e-3
        ct = np.where(past_end, self.traj.lateral_from_end_line(pose[:, 0], pose[:, 1]), ct)
        dist_end = np.hypot(pose[:, 0] - self.end[0], pose[:, 1] - self.end[1])
        ref_done = self.follower.tau >= self.traj.total_time - 1e-9
        near_end = dist_end < self.finish_tol

        reward = cfg.progress * (progress - self.progress) / self.traj.length
        reward -= cfg.time_per_baseline * dt / self.traj.total_time
        reward -= cfg.cross_track_per_s * (ct / 0.1) ** 2 * dt
        reward -= cfg.action_change * np.sum((a - self.prev_action) ** 2, axis=1)
        self.progress = progress
        self.prev_action = a
        self.ep_max_ct = np.maximum(self.ep_max_ct, ct)

        success = ref_done & (near_end | past_end)
        end_error = np.where(past_end & ~near_end, ct, dist_end)
        failure = ct > self.ct_limit
        reward += np.where(success, cfg.success_bonus - cfg.end_error_per_m * end_error, 0.0)
        reward -= np.where(failure & ~success, cfg.failure, 0.0)
        terminated = success | failure
        truncated = ~terminated & (self.follower.time >= self.max_time)
        self.ep_return += reward

        done = terminated | truncated
        info = {
            "success": success,
            "failure": failure & ~success,
            "episode_time": self.follower.time.copy(),
            "episode_return": self.ep_return.copy(),
            "max_cross_track": self.ep_max_ct.copy(),
            "end_error": np.where(success, end_error, np.nan),
            "done": done,
        }
        if done.any():
            info["final_observation"] = self._observe()
            self._reset(done)
        return self._observe(), reward, terminated, truncated, info

    # ---- 觀測 ---------------------------------------------------------------
    def _observe(self) -> np.ndarray:
        f = self.follower
        est = f.odometry.pose
        h = est[:, 2]
        t = self.traj
        tau = f.tau
        obs = np.empty((self.n, OBS_DIM))
        i = 0

        ref = t.sample(tau)
        ex, ey = field_to_robot(ref.x - est[:, 0], ref.y - est[:, 1], h)
        obs[:, i:i + 3] = np.column_stack([ex, ey, wrap_rad(ref.heading - h)])
        i += 3

        mvx, mvy, mw = chassis_from_modules(self.plant.module_speed, self.plant.module_angle,
                                            self.params.module_positions)
        obs[:, i:i + 3] = np.column_stack([mvx / 4.0, mvy / 4.0, mw / 4.0])
        i += 3

        rvx, rvy = field_to_robot(ref.vx, ref.vy, h)
        obs[:, i:i + 3] = np.column_stack([rvx / 4.0, rvy / 4.0, ref.omega / 4.0])
        i += 3

        speeds = []
        for la in LOOKAHEAD_S:
            r = t.sample(tau + la)
            lx, ly = field_to_robot(r.x - est[:, 0], r.y - est[:, 1], h)
            obs[:, i:i + 2] = np.column_stack([lx, ly]) / 2.0
            i += 2
            speeds.append(np.hypot(r.vx, r.vy) / 4.0)
        obs[:, i:i + len(LOOKAHEAD_S)] = np.column_stack(speeds)
        i += len(LOOKAHEAD_S)

        obs[:, i] = (t.total_time - tau) / t.total_time
        obs[:, i + 1] = self.plant.battery_voltage / 12.0 - 1.0
        i += 2
        obs[:, i:i + ACTION_DIM] = self.prev_action
        return obs


def evaluate_policy(env: SwerveAutoEnv, policy, episodes: int) -> dict:
    """跑到收集滿 episodes 個結束的 episode；policy(obs) → action。回傳平均指標。"""
    obs = env.reset()
    results = {k: [] for k in ("success", "episode_time", "episode_return", "max_cross_track", "end_error")}
    while len(results["success"]) < episodes:
        obs, _, _, _, info = env.step(policy(obs))
        for i in np.nonzero(info["done"])[0]:
            for k in results:
                results[k].append(info[k][i])
    summary = {k: np.array(v[:episodes], dtype=np.float64) for k, v in results.items()}
    succ = summary["success"].astype(bool)
    return {
        "episodes": episodes,
        "success_rate": float(succ.mean()),
        "mean_time_success": float(summary["episode_time"][succ].mean()) if succ.any() else float("nan"),
        "mean_return": float(summary["episode_return"].mean()),
        "max_cross_track_p95": float(np.percentile(summary["max_cross_track"], 95)),
        "mean_end_error_success": float(np.nanmean(summary["end_error"][succ])) if succ.any() else float("nan"),
    }
