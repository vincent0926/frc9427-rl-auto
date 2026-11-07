"""領域隨機化 (設計文件 §6)：每個 episode 重抽物理與感測參數。

原則：以理論值 / 程式碼常數為中心，範圍放寬，取代實機校正。策略要在整個範圍都表現好，
就不需要知道真實參數。scale ∈ [0, 1] 可以把範圍縮小 (課程學習：先窄後寬；0 = 全部用預設值)。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .params import DEFAULT_PARAMS, RobotParams

# 名稱: (下限, 上限)；物理參數對應 RealisticPlant.PER_ENV_KEYS，感測參數對應 sensors 物件的屬性
def physics_ranges(p: RobotParams = DEFAULT_PARAMS) -> dict[str, tuple[float, float]]:
    return {
        "friction_coefficient": (0.6, 1.2),
        "mass_kg": (48.0, 58.0),
        "cg_height_m": (0.15, 0.35),
        "moi_kg_m2": (0.6 * p.moi_kg_m2, 1.4 * p.moi_kg_m2),
        "gear_efficiency": (0.80, 0.97),
        "wheel_equivalent_mass_kg": (0.4, 1.6),
        "viscous_friction_n_per_mps": (0.0, 2.0 * p.viscous_friction_n_per_mps),  # kV 多 0–10%
        "coulomb_friction_n": (0.5 * p.coulomb_friction_n, 1.5 * p.coulomb_friction_n),
        "battery_open_circuit_v": (12.0, 12.8),
        "battery_resistance_ohm": (0.015, 0.025),
        "wheel_radius_scale": (0.98, 1.02),
    }


SENSOR_RANGES = {
    "vision_rate_hz": (5.0, 20.0),
    "vision_latency_s": (0.03, 0.08),
    "vision_sigma_m": (0.02, 0.08),
    "vision_dropout": (0.0, 0.5),
    "gyro_drift_rad_per_sqrt_s": (0.0, 0.004),
    "command_delay_cycles": (0, 3),  # 整數，含上限
}

START_ERROR = {"xy_m": 0.05, "heading_rad": np.radians(2.0)}


@dataclass
class Randomizer:
    rng: np.random.Generator
    scale: float = 1.0  # 0 = 不隨機 (全部用中心值)，1 = 設計文件的完整範圍
    params: RobotParams = DEFAULT_PARAMS

    def _uniform(self, low, high, center, n):
        lo = center + (low - center) * self.scale
        hi = center + (high - center) * self.scale
        return self.rng.uniform(lo, hi, n)

    def sample_physics(self, n: int) -> dict[str, np.ndarray]:
        p = self.params
        centers = {
            "friction_coefficient": p.friction_coefficient,
            "mass_kg": p.mass_kg,
            "cg_height_m": p.cg_height_m,
            "moi_kg_m2": p.moi_kg_m2,
            "gear_efficiency": p.gear_efficiency,
            "wheel_equivalent_mass_kg": p.wheel_equivalent_mass_kg,
            "viscous_friction_n_per_mps": p.viscous_friction_n_per_mps,
            "coulomb_friction_n": p.coulomb_friction_n,
            "battery_open_circuit_v": p.battery_open_circuit_v,
            "battery_resistance_ohm": p.battery_resistance_ohm,
            "wheel_radius_scale": 1.0,
        }
        return {k: self._uniform(lo, hi, centers[k], n) for k, (lo, hi) in physics_ranges(p).items()}

    def sample_sensors(self, n: int, defaults) -> dict[str, np.ndarray]:
        out = {}
        for k, (lo, hi) in SENSOR_RANGES.items():
            center = float(np.mean(getattr(defaults, k)))
            if k == "command_delay_cycles":
                lo_s = int(np.floor(center + (lo - center) * self.scale))
                hi_s = int(np.ceil(center + (hi - center) * self.scale))
                out[k] = self.rng.integers(lo_s, hi_s + 1, n)
            else:
                out[k] = self._uniform(lo, hi, center, n)
        return out

    def sample_start_offset(self, n: int) -> np.ndarray:
        xy = self.rng.uniform(-1, 1, (n, 2)) * START_ERROR["xy_m"] * self.scale
        h = self.rng.uniform(-1, 1, n) * START_ERROR["heading_rad"] * self.scale
        return np.column_stack([xy, h])


def apply_physics(plant, values: dict[str, np.ndarray], mask: np.ndarray):
    """把抽到的物理參數寫進 RealisticPlant 的每環境陣列 (只改 mask 的環境)。"""
    attr = {
        "mass_kg": "mass", "moi_kg_m2": "moi", "friction_coefficient": "mu", "cg_height_m": "cg_height",
        "wheel_equivalent_mass_kg": "wheel_mass", "gear_efficiency": "efficiency",
        "coulomb_friction_n": "coulomb", "viscous_friction_n_per_mps": "viscous",
        "battery_open_circuit_v": "v_oc", "battery_resistance_ohm": "r_batt", "wheel_radius_scale": "wheel_scale",
    }
    for key, value in values.items():
        getattr(plant, attr[key])[mask] = value[mask]


def apply_sensors(follower, values: dict[str, np.ndarray], mask: np.ndarray):
    """把抽到的感測參數寫進 PathFollower 的相機、陀螺儀、命令延遲。"""
    if follower.camera is not None:
        follower.camera.period[mask] = 1.0 / values["vision_rate_hz"][mask]
        follower.camera.latency[mask] = values["vision_latency_s"][mask]
        follower.camera.sigma[mask] = values["vision_sigma_m"][mask]
        follower.camera.dropout[mask] = values["vision_dropout"][mask]
    follower.gyro.strength[mask] = values["gyro_drift_rad_per_sqrt_s"][mask]
    follower.delay.delay[mask] = np.clip(values["command_delay_cycles"][mask], 0, follower.delay.size - 1)
