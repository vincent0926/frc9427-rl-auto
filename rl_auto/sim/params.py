"""機器人參數：全部取自 FRC9427swerve 的 Constants.java 與 PathPlanner 設定，或由馬達規格推算。

內部一律 SI 單位 (m, s, kg, V, rad)。不需實機數據；不確定的參數由領域隨機化處理 (見 docs/spec/rl-sim-design.md §6)。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

INCHES_TO_METERS = 0.0254


@dataclass(frozen=True)
class RobotParams:
    # ---- 幾何 (Constants.Swerve) ----
    track_width_m: float = 22.75 * INCHES_TO_METERS  # 左右輪距
    wheel_base_m: float = 20.75 * INCHES_TO_METERS  # 前後軸距
    wheel_radius_m: float = 2.0 * INCHES_TO_METERS
    drive_gear_ratio: float = 5.27  # SDS MK5n L3
    steer_gear_ratio: float = 287.0 / 11.0

    # ---- 質量與摩擦 ----
    mass_kg: float = 111.5 * 0.453592
    moi_kg_m2: float = 6.883  # PathPlanner settings.json 的 robotMOI
    friction_coefficient: float = 0.90

    # ---- 驅動馬達 (Kraken X60，FOC 關閉；規格以 datasheet 為準) ----
    motor_free_rps_at_12v: float = 100.0
    motor_stall_torque_nm: float = 7.09
    motor_stall_current_a: float = 366.0
    drive_stator_limit_a: float = 60.0
    drive_ks_v: float = 0.18  # Constants.Swerve.DriveS

    # ---- 控制 ----
    cycle_time_s: float = 0.02  # Constants.CycleTimeSec
    joystick_max_speed_mps: float = 4.0  # MaxDriveVelocityMps (只給遙控用)
    speed_limit_design_voltage: float = 11.0  # SpeedLimitDesignVoltage
    speed_limit_headroom: float = 0.85  # SpeedLimitHeadroom
    module_hold_speed_mps: float = 0.02  # ModuleHoldSpeedMps
    pp_translation_kp: float = 5.0  # Swerve.configureAutoBuilder
    pp_rotation_kp: float = 5.0

    # ---- 推算值 ----
    @property
    def wheel_circumference_m(self) -> float:
        return 2.0 * math.pi * self.wheel_radius_m

    @property
    def drive_volts_per_mps(self) -> float:
        """每 1 m/s 需要的反電動勢 (Constants.Swerve.DriveVoltsPerMps)。"""
        free_rps_per_volt = self.motor_free_rps_at_12v / 12.0
        return self.drive_gear_ratio / (free_rps_per_volt * self.wheel_circumference_m)

    @property
    def free_speed_mps(self) -> float:
        return self.motor_free_rps_at_12v / self.drive_gear_ratio * self.wheel_circumference_m

    @property
    def max_module_speed_mps(self) -> float:
        """desaturate 用的模組速度上限 (Constants.Swerve.MaxModuleSpeedMps)。"""
        return (
            (self.speed_limit_design_voltage - self.drive_ks_v)
            * self.speed_limit_headroom
            / self.drive_volts_per_mps
        )

    @property
    def module_positions(self) -> np.ndarray:
        """模組位置 (4, 2)，機器人座標，順序 FL, FR, BL, BR (同 Swerve.modulePositions)。"""
        hx = self.wheel_base_m * 0.5
        hy = self.track_width_m * 0.5
        return np.array([[hx, hy], [hx, -hy], [-hx, hy], [-hx, -hy]], dtype=np.float64)


DEFAULT_PARAMS = RobotParams()
