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

    def set_motion(self, field_velocity, mask=None):
        """讓車子一開始就在動 (接在另一條路線後面)：設定場地座標速度 (N, 3)，模組角度與輪速跟著對齊。"""
        mask = np.ones(self.n_envs, dtype=bool) if mask is None else mask
        angle, speed = _modules_for_motion(self.params, field_velocity, self.pose[:, 2])
        self.module_angle[mask] = angle[mask]
        self.module_speed[mask] = speed[mask]

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


class RealisticPlant:
    """真實模型 (設計文件 §4)：TalonFX 速度迴路 + Kraken 馬達 + 輪胎滑移 + 載重轉移 + 電池壓降。

    - 車體：平面剛體，場地座標速度 (vx, vy) 與角速度 ω
    - 每個模組：轉向角 (一階 + 轉速上限)、輪面線速度 v_w (有自己的動態，打滑時跟地面速度不同)
    - 輪胎：接觸點滑移 → 線性摩擦力，摩擦圓 μ·N_i 飽和
    - 輪速更新對輪胎剛性項用隱式 (backward Euler)：輪子等效質量很小、輪胎很硬，顯式會發散
    - 編碼器量的是輪速，所以打滑時 drive_position (里程計) 會跟真實位移不同

    每個環境可以有不同參數：per_env 字典覆寫 (純量或形狀 (N,) 的陣列)，鍵名見 PER_ENV_KEYS。
    """

    substeps: int = 10
    PER_ENV_KEYS = (
        "mass_kg", "moi_kg_m2", "friction_coefficient", "cg_height_m", "wheel_equivalent_mass_kg",
        "gear_efficiency", "coulomb_friction_n", "viscous_friction_n_per_mps",
        "battery_open_circuit_v", "battery_resistance_ohm", "wheel_radius_scale",
    )

    def __init__(self, n_envs: int, params: RobotParams = DEFAULT_PARAMS, per_env: dict | None = None):
        self.params = params
        self.n_envs = n_envs
        self.positions = params.module_positions  # (4, 2)
        p = params
        overrides = per_env or {}
        unknown = set(overrides) - set(self.PER_ENV_KEYS)
        if unknown:
            raise KeyError(f"未知的 per_env 參數：{sorted(unknown)}")

        def env_param(name, default):
            value = overrides.get(name, default)
            return np.broadcast_to(np.asarray(value, dtype=np.float64), (n_envs,)).copy()

        self.mass = env_param("mass_kg", p.mass_kg)
        self.moi = env_param("moi_kg_m2", p.moi_kg_m2)
        self.mu = env_param("friction_coefficient", p.friction_coefficient)
        self.cg_height = env_param("cg_height_m", p.cg_height_m)
        self.wheel_mass = env_param("wheel_equivalent_mass_kg", p.wheel_equivalent_mass_kg)
        self.efficiency = env_param("gear_efficiency", p.gear_efficiency)
        self.coulomb = env_param("coulomb_friction_n", p.coulomb_friction_n)
        self.viscous = env_param("viscous_friction_n_per_mps", p.viscous_friction_n_per_mps)
        self.v_oc = env_param("battery_open_circuit_v", p.battery_open_circuit_v)
        self.r_batt = env_param("battery_resistance_ohm", p.battery_resistance_ohm)
        self.wheel_scale = env_param("wheel_radius_scale", 1.0)  # 實際輪徑 / 程式設定輪徑 (磨耗)

        n = n_envs
        self.pose = np.zeros((n, 3))
        self.velocity = np.zeros((n, 3))  # 場地座標 vx, vy, ω
        self.accel_robot = np.zeros((n, 2))  # 上一子步的機器人座標加速度 (載重轉移用)
        self.module_angle = np.zeros((n, 4))
        self.wheel_speed = np.zeros((n, 4))  # 真實輪面線速度
        self.drive_position = np.zeros((n, 4))  # 編碼器讀數換算的行駛距離 (用程式設定的輪徑)
        self.current = np.zeros((n, 4))  # 驅動馬達 stator 電流
        self.battery_voltage = self.v_oc.copy()
        self.slip = np.zeros((n, 4))  # 滑移速度大小 (分析用)

    # 介面同 IdealPlant -------------------------------------------------------
    @property
    def heading(self) -> np.ndarray:
        return self.pose[:, 2]

    @property
    def module_speed(self) -> np.ndarray:
        """編碼器量到的輪速 (用程式設定的輪徑換算)。"""
        return self.wheel_speed / self.wheel_scale[:, None]

    def reset(self, pose, mask=None):
        if mask is None:
            mask = np.ones(self.n_envs, dtype=bool)
        self.pose[mask] = pose[mask] if np.ndim(pose) == 2 else pose
        for arr in (self.velocity, self.accel_robot, self.module_angle, self.wheel_speed,
                    self.drive_position, self.current, self.slip):
            arr[mask] = 0.0
        self.battery_voltage[mask] = self.v_oc[mask]

    def set_motion(self, field_velocity, mask=None):
        """讓車子一開始就在動 (接在另一條路線後面)：車體速度、模組角度、輪速一致，不打滑。"""
        mask = np.ones(self.n_envs, dtype=bool) if mask is None else mask
        angle, speed = _modules_for_motion(self.params, field_velocity, self.pose[:, 2])
        self.velocity[mask] = np.asarray(field_velocity, dtype=np.float64)[mask]
        self.module_angle[mask] = angle[mask]
        self.wheel_speed[mask] = speed[mask]

    def step(self, target_angle, target_speed, dt):
        target_angle = np.asarray(target_angle, dtype=np.float64)
        target_speed = np.asarray(target_speed, dtype=np.float64)
        h = dt / self.substeps
        for _ in range(self.substeps):
            self._substep(target_angle, target_speed, h)

    # 物理 ---------------------------------------------------------------------
    def _substep(self, target_angle, target_speed, h):
        p = self.params
        g = 9.81

        # 1. 轉向：一階追目標，轉速上限 = 馬達自由轉速 / 轉向齒比
        steer_max = p.motor_free_rps_at_12v / p.steer_gear_ratio * 2.0 * np.pi
        err = wrap_rad(target_angle - self.module_angle)
        step = np.clip(err / p.steer_time_constant_s, -steer_max, steer_max) * h
        self.module_angle = self.module_angle + np.clip(step, -np.abs(err), np.abs(err))

        # 2. TalonFX VelocityVoltage：目標與回授都是程式設定輪徑下的速度
        v_cmd = (p.drive_ks_v * np.sign(target_speed) + p.drive_kv_v_per_mps * target_speed
                 + p.drive_kp_v_per_mps * (target_speed - self.module_speed))
        v_bat = self.battery_voltage[:, None]
        v_cmd = np.clip(v_cmd, -v_bat, v_bat)

        # 3. 馬達電流 (反電動勢依真實馬達轉速)、stator 限流
        back_emf = self.module_speed * p.drive_volts_per_mps
        current = np.clip((v_cmd - back_emf) / p.motor_resistance_ohm,
                          -p.drive_stator_limit_a, p.drive_stator_limit_a)
        self.current = current
        f_motor = current * p.force_per_amp_n / self.wheel_scale[:, None] * self.efficiency[:, None]
        f_motor = f_motor - self.coulomb[:, None] * np.tanh(self.wheel_speed / 0.01)

        # 4. 電池：供電電流 ≈ stator 電流 × 佔空比。電壓低通 10 ms：直接代入會在子步間來回震盪
        #    (電壓高 → 電流大 → 電壓掉 → 電流變小 …)，實際電池與線路也有這種緩衝。
        supply = np.sum(np.abs(current * v_cmd), axis=1) / np.maximum(self.battery_voltage, 1.0)
        target_v = np.maximum(self.v_oc - self.r_batt * supply, 6.0)
        self.battery_voltage += (target_v - self.battery_voltage) * min(h / 0.01, 1.0)

        # 5. 載重轉移 (用上一子步加速度，機器人座標)
        ax, ay = self.accel_robot[:, 0:1], self.accel_robot[:, 1:2]
        sx = np.sign(self.positions[:, 0])  # 前 +1 / 後 -1
        sy = np.sign(self.positions[:, 1])  # 左 +1 / 右 -1
        m = self.mass[:, None]
        hcg = self.cg_height[:, None]
        normal = (m * g / 4.0 - m * ax * hcg / (2.0 * p.wheel_base_m) * sx
                  - m * ay * hcg / (2.0 * p.track_width_m) * sy)
        f_max = self.mu[:, None] * np.maximum(normal, 0.0)

        # 6. 輪胎：接觸點對地速度 (機器人座標) → 輪子座標 (縱向、側向)
        heading = self.pose[:, 2]
        cos_h, sin_h = np.cos(heading), np.sin(heading)
        rvx = cos_h * self.velocity[:, 0] + sin_h * self.velocity[:, 1]
        rvy = -sin_h * self.velocity[:, 0] + cos_h * self.velocity[:, 1]
        w = self.velocity[:, 2:3]
        cvx = rvx[:, None] - w * self.positions[:, 1]
        cvy = rvy[:, None] + w * self.positions[:, 0]
        cos_a, sin_a = np.cos(self.module_angle), np.sin(self.module_angle)
        v_long = cvx * cos_a + cvy * sin_a
        v_lat = -cvx * sin_a + cvy * cos_a

        stiffness = self.mu[:, None] * (m * g / 4.0) / p.tire_slip_saturation_mps
        mw = self.wheel_mass[:, None]
        b = self.viscous[:, None]
        # 隱式輪速 (線性區)：mw·(v' − v)/h = F_motor − b·v' − C·(v' − v_long)
        v_lin = (self.wheel_speed + h / mw * (f_motor + stiffness * v_long)) / (1.0 + h * (b + stiffness) / mw)
        f_long = stiffness * (v_lin - v_long)
        f_lat = -stiffness * v_lat
        mag = np.hypot(f_long, f_lat)
        scale = np.where(mag > f_max, f_max / np.maximum(mag, 1e-9), 1.0)
        f_long = f_long * scale
        f_lat = f_lat * scale
        saturated = scale < 1.0
        # 摩擦圓飽和時地面力有上限、不再是硬的，輪速用顯式更新 (會空轉或鎖死)
        v_sat = self.wheel_speed + h / mw * (f_motor - b * self.wheel_speed - f_long)
        self.wheel_speed = np.where(saturated, v_sat, v_lin)
        self.slip = np.hypot(self.wheel_speed - v_long, v_lat)

        # 7. 車體：輪胎力 (機器人座標) 加總 → 場地座標加速度
        fx = f_long * cos_a - f_lat * sin_a
        fy = f_long * sin_a + f_lat * cos_a
        force_x = fx.sum(axis=1)
        force_y = fy.sum(axis=1)
        torque = (self.positions[:, 0] * fy - self.positions[:, 1] * fx).sum(axis=1)
        self.accel_robot[:, 0] = force_x / self.mass
        self.accel_robot[:, 1] = force_y / self.mass
        afx, afy = robot_to_field(self.accel_robot[:, 0], self.accel_robot[:, 1], heading)

        # 半隱式 Euler：先更新速度再更新位置
        self.velocity[:, 0] += afx * h
        self.velocity[:, 1] += afy * h
        self.velocity[:, 2] += torque / self.moi * h
        self.pose[:, 0] += self.velocity[:, 0] * h
        self.pose[:, 1] += self.velocity[:, 1] * h
        self.pose[:, 2] = wrap_rad(self.pose[:, 2] + self.velocity[:, 2] * h)

        # 編碼器：輪子轉了多少 (用程式設定的輪徑換算)
        self.drive_position = self.drive_position + self.module_speed * h


def _modules_for_motion(params, field_velocity, heading):
    """場地座標速度 → 各模組角度與輪面速度 (不做反轉最佳化)。"""
    from .robot_software import calculate_module_targets, field_to_robot

    field_velocity = np.asarray(field_velocity, dtype=np.float64)
    rvx, rvy = field_to_robot(field_velocity[:, 0], field_velocity[:, 1], heading)
    speed, angle = calculate_module_targets(rvx, rvy, field_velocity[:, 2], params.module_positions)
    return angle, speed
