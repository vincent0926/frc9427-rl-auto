"""機器人軟體的 Python 移植 (向量化)。

對應 FRC9427swerve：
- SwerveMath.java：座標轉換、模組目標、desaturate、弧線位移、wrapRad
- SwerveModule.setTarget：模組最佳化 (反轉 180°、cos 縮放)
- Swerve.operate() 的 followingPath 分支、Swerve.updateOdometry()
- PPHolonomicDriveController：前饋 + P 回授

所有函式都接受批次陣列：前導維度 N 是環境數，模組維度固定為 4。
移植必須跟 Java 行為一致；改 Java 時要同步改這裡並更新 tests/test_robot_software.py。
"""

from __future__ import annotations

import numpy as np

from .params import DEFAULT_PARAMS, RobotParams

PI = np.pi
TAU = 2.0 * np.pi
HALF_PI = 0.5 * np.pi


def wrap_rad(radians):
    """包到 [-π, π)，同 SwerveMath.wrapRad。"""
    return (np.asarray(radians) + PI) % TAU - PI


def field_to_robot(field_x, field_y, heading_rad):
    cos = np.cos(heading_rad)
    sin = np.sin(heading_rad)
    return cos * field_x + sin * field_y, -sin * field_x + cos * field_y


def robot_to_field(robot_x, robot_y, heading_rad):
    cos = np.cos(heading_rad)
    sin = np.sin(heading_rad)
    return cos * robot_x - sin * robot_y, sin * robot_x + cos * robot_y


def calculate_module_targets(robot_vx, robot_vy, omega, module_positions, center_of_rotation=(0.0, 0.0)):
    """機器人座標速度 → 各模組速度與角度。輸入 (N,)，輸出 (N, 4)。"""
    robot_vx = np.asarray(robot_vx, dtype=np.float64)[..., None]
    robot_vy = np.asarray(robot_vy, dtype=np.float64)[..., None]
    omega = np.asarray(omega, dtype=np.float64)[..., None]
    radius_x = module_positions[:, 0] - center_of_rotation[0]
    radius_y = module_positions[:, 1] - center_of_rotation[1]
    wheel_x = robot_vx - omega * radius_y
    wheel_y = robot_vy + omega * radius_x
    return np.hypot(wheel_x, wheel_y), np.arctan2(wheel_y, wheel_x)


def desaturate(speeds, max_speed):
    """任一模組超過 max_speed 時，四個模組依同一比例縮小。回傳 (縮放後速度, 比例)。"""
    speeds = np.asarray(speeds, dtype=np.float64)
    highest = np.max(np.abs(speeds), axis=-1, keepdims=True)
    scale = np.where(highest > max_speed, max_speed / np.maximum(highest, 1e-12), 1.0)
    return speeds * scale, scale[..., 0]


def optimize_module(wanted_angle, wanted_speed, current_angle):
    """SwerveModule.setTarget：轉超過 90° 就反轉方向，速度再乘 |cos(角度誤差)|。

    回傳 (目標角度, 目標速度)。目標角度是「目前角度 + 差值」，不包回 [-π, π)，同 Java。
    """
    delta = wrap_rad(np.asarray(wanted_angle) - current_angle)
    speed = np.asarray(wanted_speed, dtype=np.float64)
    flip = np.abs(delta) > HALF_PI
    delta = np.where(flip, wrap_rad(delta + PI), delta)
    speed = np.where(flip, -speed, speed)
    speed = speed * np.abs(np.cos(delta))
    return current_angle + delta, speed


def arc_displacement(previous_angle, angle, distance):
    """SwerveMath.arcDisplacement：模組沿圓弧從 previous_angle 轉到 angle、走 distance 的位移。"""
    previous_angle = np.asarray(previous_angle, dtype=np.float64)
    angle = np.asarray(angle, dtype=np.float64)
    distance = np.asarray(distance, dtype=np.float64)
    delta = wrap_rad(angle - previous_angle)
    straight = np.abs(delta) < 1e-5
    safe_delta = np.where(straight, 1.0, delta)
    radius = distance / safe_delta
    arc_x = np.cos(angle - HALF_PI) * radius - np.cos(previous_angle - HALF_PI) * radius
    arc_y = np.sin(angle - HALF_PI) * radius - np.sin(previous_angle - HALF_PI) * radius
    out_x = np.where(straight, np.cos(angle) * distance, arc_x)
    out_y = np.where(straight, np.sin(angle) * distance, arc_y)
    return out_x, out_y


def pp_holonomic_controller(ref_x, ref_y, ref_heading, ref_vx, ref_vy, ref_omega,
                            pose_x, pose_y, pose_heading,
                            translation_kp, rotation_kp):
    """PPHolonomicDriveController：場地座標前饋 + P 回授 (kI = kD = 0)。

    Java 端回傳機器人座標速度，Swerve.driveRobotRelative 又用同一個朝向轉回場地座標，
    所以這裡直接回傳場地座標速度 (field_vx, field_vy, omega)。
    """
    field_vx = ref_vx + translation_kp * (ref_x - pose_x)
    field_vy = ref_vy + translation_kp * (ref_y - pose_y)
    omega = ref_omega + rotation_kp * wrap_rad(ref_heading - pose_heading)
    return field_vx, field_vy, omega


class SwerveSoftware:
    """Swerve.operate() 跟隨路線分支 + SwerveModule.setTarget 的批次版本。"""

    def __init__(self, n_envs: int, params: RobotParams = DEFAULT_PARAMS):
        self.params = params
        self.n_envs = n_envs
        self.module_positions = params.module_positions
        # SwerveModule 記住的上一個目標角度 (保持模組時沿用)
        self.target_angle = np.zeros((n_envs, 4))

    def reset(self, mask=None):
        if mask is None:
            self.target_angle[:] = 0.0
        else:
            self.target_angle[mask] = 0.0

    def follow_path(self, field_vx, field_vy, omega, heading, module_angles):
        """場地座標速度命令 → 各模組 (目標角度, 目標速度)，皆為 (N, 4)。

        module_angles：各模組目前實際角度 (N, 4)，optimize 需要。
        """
        p = self.params
        robot_vx, robot_vy = field_to_robot(field_vx, field_vy, heading)
        omega = np.asarray(omega, dtype=np.float64)

        # 速度與角速度都接近 0 → holdModules() → SwerveModule.stop()：目標角度 = 目前實際角度、速度 0
        idle =(robot_vx ** 2 + robot_vy ** 2 < 1e-6) & (np.abs(omega) < 1e-4)

        speeds, angles = calculate_module_targets(robot_vx, robot_vy, omega, self.module_positions)
        speeds, _ = desaturate(speeds, p.max_module_speed_mps)

        # 模組速度太小 → 角度維持上一個目標、速度 0 (避免低速時模組亂轉)
        slow = np.abs(speeds) < p.module_hold_speed_mps
        wanted_angle = np.where(slow, self.target_angle, angles)
        wanted_speed = np.where(slow, 0.0, speeds)

        target_angle, target_speed = optimize_module(wanted_angle, wanted_speed, module_angles)

        target_angle = np.where(idle[..., None], module_angles, target_angle)
        target_speed = np.where(idle[..., None], 0.0, target_speed)
        self.target_angle = target_angle
        return target_angle, target_speed


class Odometry:
    """Swerve.updateOdometry() 的批次版本：各模組弧線位移 → 場地座標 → 四輪平均。"""

    def __init__(self, n_envs: int, params: RobotParams = DEFAULT_PARAMS):
        self.module_positions = params.module_positions
        self.pose = np.zeros((n_envs, 3))  # x, y, heading
        self.module_field = np.zeros((n_envs, 4, 2))
        self.prev_drive_position = np.zeros((n_envs, 4))
        self.prev_module_angle = np.zeros((n_envs, 4))

    def reset(self, pose, drive_position, module_angle, mask=None):
        """設定位姿 (同 Swerve.resetPose)，並記下模組目前讀數當作下一次位移的起點。"""
        if mask is None:
            mask = np.ones(self.pose.shape[0], dtype=bool)
        self.pose[mask] = pose[mask] if np.ndim(pose) == 2 else pose
        self.prev_drive_position[mask] = drive_position[mask]
        self.prev_module_angle[mask] = module_angle[mask]
        self._refit(mask)

    def _refit(self, mask):
        heading = self.pose[mask, 2][:, None]
        off_x, off_y = robot_to_field(self.module_positions[:, 0], self.module_positions[:, 1], heading)
        self.module_field[mask, :, 0] = self.pose[mask, 0][:, None] + off_x
        self.module_field[mask, :, 1] = self.pose[mask, 1][:, None] + off_y

    def update(self, drive_position, module_angle, heading):
        """drive_position：各模組累積行駛距離 (m)；module_angle：模組角度 (機器人座標)；heading：陀螺儀朝向。"""
        distance = drive_position - self.prev_drive_position
        dx, dy = arc_displacement(self.prev_module_angle, module_angle, distance)
        self.prev_drive_position = drive_position.copy()
        self.prev_module_angle = module_angle.copy()

        # 同 Java (2026-11-05 修正)：位移用週期中間的朝向轉到場地座標
        last = self.pose[:, 2]
        mid = (last + wrap_rad(heading - last) * 0.5)[:, None]
        h = heading[:, None]
        fdx, fdy = robot_to_field(dx, dy, mid)
        self.module_field[..., 0] += fdx
        self.module_field[..., 1] += fdy
        off_x, off_y = robot_to_field(self.module_positions[:, 0], self.module_positions[:, 1], h)
        self.pose[:, 0] = np.mean(self.module_field[..., 0] - off_x, axis=-1)
        self.pose[:, 1] = np.mean(self.module_field[..., 1] - off_y, axis=-1)
        self.pose[:, 2] = heading
        self._refit(np.ones(self.pose.shape[0], dtype=bool))
        return self.pose


def chassis_from_modules(speeds, angles, module_positions):
    """各模組實際速度與角度 → 機器人座標底盤速度 (vx, vy, ω)，最小平方 (同 Swerve.getRobotRelativeSpeeds)。"""
    a = np.zeros((8, 3))
    for i, (px, py) in enumerate(module_positions):
        a[2 * i] = [1.0, 0.0, -py]
        a[2 * i + 1] = [0.0, 1.0, px]
    pinv = np.linalg.pinv(a)
    wheel = np.empty(speeds.shape[:-1] + (8,))
    wheel[..., 0::2] = speeds * np.cos(angles)
    wheel[..., 1::2] = speeds * np.sin(angles)
    out = wheel @ pinv.T
    return out[..., 0], out[..., 1], out[..., 2]
