"""底盤物理 (受控體)。

目前只有理想模型 IdealPlant：模組瞬間到達目標角度與速度、不打滑、無延遲。
用途是健全性檢查 (設計文件 §11 第 3 步)：機器人軟體 + 理想物理跑 PathPlanner 路線應幾乎完美跟隨。
之後的真實模型 (馬達、輪胎打滑、載重轉移) 要提供同樣的介面，執行器 (rollout.py) 不用改。

介面 (批次，N 個環境)：
- reset(pose, mask)：設定真實位姿 (N, 3)
- step(target_angle, target_speed, dt)：套用模組命令 (N, 4) 並前進 dt
- 屬性：pose (真實 x, y, heading)、module_angle、module_speed、drive_position (累積行駛距離)、heading (陀螺儀)
"""

from __future__ import annotations

import numpy as np

from .params import DEFAULT_PARAMS, RobotParams
from .robot_software import robot_to_field, wrap_rad


class IdealPlant:
    def __init__(self, n_envs: int, params: RobotParams = DEFAULT_PARAMS):
        self.params = params
        self.n_envs = n_envs
        positions = params.module_positions
        # 由四個模組速度向量最小平方解出底盤速度：[vx, vy, ω] = pinv(A) · 模組速度
        a = np.zeros((8, 3))
        for i, (px, py) in enumerate(positions):
            a[2 * i] = [1.0, 0.0, -py]
            a[2 * i + 1] = [0.0, 1.0, px]
        self._forward = np.linalg.pinv(a)  # (3, 8)

        self.pose = np.zeros((n_envs, 3))
        self.robot_velocity = np.zeros((n_envs, 3))  # 機器人座標 vx, vy, ω
        self.module_angle = np.zeros((n_envs, 4))
        self.module_speed = np.zeros((n_envs, 4))
        self.drive_position = np.zeros((n_envs, 4))

    @property
    def heading(self) -> np.ndarray:
        return self.pose[:, 2]

    def reset(self, pose, mask=None):
        if mask is None:
            mask = np.ones(self.n_envs, dtype=bool)
        self.pose[mask] = pose[mask] if np.ndim(pose) == 2 else pose
        self.robot_velocity[mask] = 0.0
        self.module_speed[mask] = 0.0
        self.module_angle[mask] = 0.0
        self.drive_position[mask] = 0.0

    substeps: int = 10

    def step(self, target_angle, target_speed, dt):
        """模組在一個週期內從目前角度「等速轉」到目標角度 (理想但連續)，輪速立即到目標。

        若角度瞬間跳到目標，會跟 Java 里程計的「弧線」假設 (週期內模組從舊角度轉到新角度) 不一致，
        產生理想模型本身造成的假誤差。
        """
        start_angle = self.module_angle
        delta = wrap_rad(np.asarray(target_angle, dtype=np.float64) - start_angle)
        self.module_speed = np.asarray(target_speed, dtype=np.float64).copy()
        h = dt / self.substeps
        for k in range(self.substeps):
            angle = start_angle + delta * ((k + 0.5) / self.substeps)
            wheel = np.empty((self.n_envs, 8))
            wheel[:, 0::2] = self.module_speed * np.cos(angle)
            wheel[:, 1::2] = self.module_speed * np.sin(angle)
            self.robot_velocity = wheel @ self._forward.T
            self._integrate(h)
        self.module_angle = start_angle + delta
        self.drive_position = self.drive_position + self.module_speed * dt

    def _integrate(self, dt):
        """定速度扭轉積分 (pose exponential)：dt 內底盤速度固定，走圓弧。"""
        vx, vy, w = self.robot_velocity.T
        theta = w * dt
        small = np.abs(theta) < 1e-9
        safe = np.where(small, 1.0, theta)
        sin_t = np.where(small, 1.0 - theta**2 / 6.0, np.sin(safe) / safe)
        cos_t = np.where(small, theta / 2.0, (1.0 - np.cos(safe)) / safe)
        dx_robot = (vx * sin_t - vy * cos_t) * dt
        dy_robot = (vx * cos_t + vy * sin_t) * dt
        dx, dy = robot_to_field(dx_robot, dy_robot, self.pose[:, 2])
        self.pose[:, 0] += dx
        self.pose[:, 1] += dy
        self.pose[:, 2] = wrap_rad(self.pose[:, 2] + theta)
