"""閉迴路執行器：PathPlanner 參考 → (時間縮放 + 殘差) → 機器人軟體 → 物理 → 里程計。

這是之後 Gymnasium 環境 (env.py) 的核心迴圈。順序對應 roboRIO 每個 20 ms 週期：
1. 讀感測 → 里程計更新 (Swerve.update)
2. 推進參考時間 τ += s·dt，取樣參考
3. PP 控制器用「估計位姿」算速度命令，加上殘差
4. Swerve.operate → 模組命令
5. 物理前進一個週期
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .chassis import IdealPlant
from .params import DEFAULT_PARAMS, RobotParams
from .robot_software import Odometry, SwerveSoftware, pp_holonomic_controller, wrap_rad
from .trajectory import Trajectory


@dataclass
class RolloutResult:
    finish_time: np.ndarray  # (N,) 到終點的時間 (未完成為 nan)
    max_cross_track: np.ndarray  # (N,) 全程最大橫向誤差 (真實位姿，m)
    end_position_error: np.ndarray  # (N,) 終點位置誤差 (m)
    end_heading_error: np.ndarray  # (N,) 終點朝向誤差 (rad)
    max_odometry_error: np.ndarray  # (N,) 里程計與真實位置的最大差距 (m)
    log: dict = field(default_factory=dict)  # 選用：逐週期紀錄 (第 0 個環境)


class PathFollower:
    def __init__(self, trajectory: Trajectory, n_envs: int = 1, params: RobotParams = DEFAULT_PARAMS, plant=None):
        self.traj = trajectory
        self.params = params
        self.n_envs = n_envs
        self.plant = plant if plant is not None else IdealPlant(n_envs, params)
        self.software = SwerveSoftware(n_envs, params)
        self.odometry = Odometry(n_envs, params)
        self.tau = np.zeros(n_envs)
        self.time = np.zeros(n_envs)

    def reset(self, start_offset=None):
        """起點 = 路線起點 (+ 選用的真實位姿誤差 (N, 3))。里程計以「路線起點」重置，同 AutoBuilder.resetPose。

        路線起點速度不為 0 (接在另一條路線後面) 時，車子一開始就以參考速度在動。
        """
        start = np.tile(self.traj.start_pose, (self.n_envs, 1))
        true_start = start + (start_offset if start_offset is not None else 0.0)
        self.plant.reset(true_start)
        ref = self.traj.sample(np.zeros(self.n_envs))
        start_velocity = np.stack([ref.vx, ref.vy, ref.omega], axis=1)
        if np.any(np.abs(start_velocity) > 1e-6):
            self.plant.set_motion(start_velocity)
        self.software.reset()
        self.odometry.reset(start, self.plant.drive_position, self.plant.module_angle)
        self.tau[:] = 0.0
        self.time[:] = 0.0

    def step(self, time_scale=1.0, residual=None):
        """前進一個控制週期。time_scale: (N,) 時間縮放 s；residual: (N, 3) 場地座標 Δvx, Δvy, Δω。"""
        dt = self.params.cycle_time_s
        est = self.odometry.update(self.plant.drive_position, self.plant.module_angle, self.plant.heading)

        self.tau = np.minimum(self.tau + np.asarray(time_scale) * dt, self.traj.total_time)
        ref = self.traj.sample(self.tau, time_scale)
        vx, vy, w = pp_holonomic_controller(
            ref.x, ref.y, ref.heading, ref.vx, ref.vy, ref.omega,
            est[:, 0], est[:, 1], est[:, 2],
            self.params.pp_translation_kp, self.params.pp_rotation_kp,
        )
        if residual is not None:
            vx = vx + residual[:, 0]
            vy = vy + residual[:, 1]
            w = w + residual[:, 2]

        angle, speed = self.software.follow_path(vx, vy, w, est[:, 2], self.plant.module_angle)
        self.plant.step(angle, speed, dt)
        self.time += dt
        return est

    def run(self, max_time=None, finish_tolerance_m=0.05, record=False) -> RolloutResult:
        """用 s = 1、無殘差跑完整條路線 (PathPlanner 基準)。"""
        self.reset()
        max_time = max_time if max_time is not None else self.traj.total_time * 1.5 + 1.0
        n = self.n_envs
        end = np.array([self.traj.x[-1], self.traj.y[-1], self.traj.heading_unwrapped[-1]])
        finish = np.full(n, np.nan)
        end_error = np.full(n, np.nan)
        end_heading = np.full(n, np.nan)
        max_ct = np.zeros(n)
        max_odo = np.zeros(n)
        log = {"t": [], "true": [], "est": [], "ref": []} if record else {}

        steps = int(np.ceil(max_time / self.params.cycle_time_s))
        for _ in range(steps):
            # 里程計估計的是「這個週期開始時」的位置，要跟同一時刻的真實位置比
            true_before = self.plant.pose.copy()
            est = self.step()
            true = self.plant.pose
            ct, progress = self.traj.cross_track(true[:, 0], true[:, 1])
            # 衝過終點 (進度已到路長) 的部分不是偏線，只算對最後一段延長線的橫向距離
            past_end = progress >= self.traj.length - 1e-3
            ct = np.where(past_end, self.traj.lateral_from_end_line(true[:, 0], true[:, 1]), ct)
            running = np.isnan(finish)
            max_ct = np.where(running, np.maximum(max_ct, ct), max_ct)
            odo_error = np.hypot(est[:, 0] - true_before[:, 0], est[:, 1] - true_before[:, 1])
            max_odo = np.where(running, np.maximum(max_odo, odo_error), max_odo)
            # 完成 = 參考時間跑完，且 (到達終點附近，或沿路線進度已過終點：終點速度不為 0 的路線會直接衝過去)
            near_end = np.hypot(true[:, 0] - end[0], true[:, 1] - end[1]) < finish_tolerance_m
            done = running & (self.tau >= self.traj.total_time) & (near_end | past_end)
            finish = np.where(done, self.time, finish)
            # 停在終點的路線：到終點的距離；衝過終點的路線：橫向偏移 (沿路線方向超過不算誤差)
            end_dist = np.where(past_end & ~near_end, ct, np.hypot(true[:, 0] - end[0], true[:, 1] - end[1]))
            end_error = np.where(done, end_dist, end_error)
            end_heading = np.where(done, np.abs(wrap_rad(true[:, 2] - end[2])), end_heading)
            if record:
                ref = self.traj.sample(self.tau[:1])
                log["t"].append(self.time[0])
                log["true"].append(true[0].copy())
                log["est"].append(est[0].copy())
                log["ref"].append([ref.x[0], ref.y[0], ref.heading[0]])
            if np.all(~np.isnan(finish)):
                break

        true = self.plant.pose
        unfinished = np.isnan(finish)
        end_error = np.where(unfinished, np.hypot(true[:, 0] - end[0], true[:, 1] - end[1]), end_error)
        end_heading = np.where(unfinished, np.abs(wrap_rad(true[:, 2] - end[2])), end_heading)
        return RolloutResult(
            finish_time=finish,
            max_cross_track=max_ct,
            end_position_error=end_error,
            end_heading_error=end_heading,
            max_odometry_error=max_odo,
            log={k: np.array(v) for k, v in log.items()},
        )
