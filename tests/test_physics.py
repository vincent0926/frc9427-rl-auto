"""真實物理模型的理論健全性檢查 (設計文件 §7)：不用實機數據，只跟理論值比。"""

import numpy as np
import pytest

from rl_auto.sim.chassis import RealisticPlant
from rl_auto.sim.params import DEFAULT_PARAMS as P
from rl_auto.sim.robot_software import Odometry, calculate_module_targets

DT = P.cycle_time_s

# 去掉損耗、電池內阻、重心高度：只剩要檢查的那一項限制
LOSSLESS = dict(viscous_friction_n_per_mps=0.0, coulomb_friction_n=0.0, battery_resistance_ohm=0.0,
                battery_open_circuit_v=12.0, cg_height_m=0.0, gear_efficiency=1.0)


def run(plant, angle, speed, seconds):
    angle = np.broadcast_to(angle, (plant.n_envs, 4))
    speed = np.broadcast_to(speed, (plant.n_envs, 4))
    for _ in range(int(round(seconds / DT))):
        plant.step(angle, speed, DT)


def make(**overrides):
    plant = RealisticPlant(1, per_env=overrides)
    plant.reset(np.zeros(3))
    return plant


def test_no_command_no_motion():
    plant = make()
    run(plant, 0.0, 0.0, 2.0)
    np.testing.assert_allclose(plant.pose[0], 0.0, atol=1e-9)
    np.testing.assert_allclose(plant.wheel_speed, 0.0, atol=1e-9)


def test_top_speed_matches_free_speed_without_losses():
    # 全電壓、無損耗：極速 = 12 V / (1.98 V·s/m) ≈ 6.06 m/s
    plant = make(**LOSSLESS)
    run(plant, 0.0, 8.0, 4.0)
    assert plant.velocity[0, 0] == pytest.approx(P.free_speed_mps, rel=0.01)


def test_top_speed_with_default_losses_is_lower():
    plant = make()
    run(plant, 0.0, 8.0, 4.0)
    assert 4.5 < plant.velocity[0, 0] < P.free_speed_mps


def test_friction_limited_acceleration():
    # 摩擦係數很低：最大加速度 ≈ μ·g
    plant = make(**LOSSLESS, friction_coefficient=0.3)
    run(plant, 0.0, 6.0, 0.3)
    accel = plant.velocity[0, 0] / 0.3
    assert accel == pytest.approx(0.3 * 9.81, rel=0.05)
    assert np.all(plant.wheel_speed > plant.velocity[0, 0])  # 輪子在空轉


def test_current_limited_acceleration():
    # 摩擦係數很高：加速度受 stator 60 A 限制 = 4 × 60 A × (每安培輪面力) / 質量
    plant = make(**LOSSLESS, friction_coefficient=2.0)
    run(plant, 0.0, 6.0, 0.2)
    # 要一起加速的質量 = 車重 + 四個輪子的等效質量
    expected = 4 * P.drive_stator_limit_a * P.force_per_amp_n / (P.mass_kg + 4 * P.wheel_equivalent_mass_kg)
    assert plant.velocity[0, 0] / 0.2 == pytest.approx(expected, rel=0.05)


def test_configured_kv_overspeeds():
    # 機器人設定的 kV 比「理論 + 5% 損耗」高，速度迴路 kP 幾乎 0 → 穩態速度比命令快約 12%
    plant = make()
    run(plant, 0.0, 2.0, 2.0)
    assert 2.15 < plant.velocity[0, 0] < 2.35


def test_matching_kv_tracks_command():
    # 若損耗剛好吃掉設定 kV 多出的部分 (19%)，速度就跟命令一樣
    excess = P.drive_kv_v_per_mps / P.drive_volts_per_mps - 1.0
    b = P.drive_volts_per_mps * excess / P.motor_resistance_ohm * P.force_per_amp_n * P.gear_efficiency
    plant = make(viscous_friction_n_per_mps=b)
    run(plant, 0.0, 2.0, 2.0)
    assert plant.velocity[0, 0] == pytest.approx(2.0, rel=0.01)


def test_load_transfer_unloads_front_wheels():
    plant = make(cg_height_m=0.3, friction_coefficient=1.0)
    run(plant, 0.0, 6.0, 0.1)
    # 往前加速：前輪 (0, 1) 載重變小、先空轉
    assert plant.slip[0, 0] > plant.slip[0, 2]
    assert plant.slip[0, 1] > plant.slip[0, 3]


def test_spin_in_place():
    plant = make()
    pos = P.module_positions
    radius = np.hypot(pos[:, 0], pos[:, 1])
    angle = np.arctan2(pos[:, 0], -pos[:, 1])
    plant.module_angle[:] = angle
    run(plant, angle, 2.0 * radius, 2.0)
    # 設定的 kV 偏高 → 轉得比命令快約 13% (同 test_configured_kv_overspeeds)
    assert 2.1 < plant.velocity[0, 2] < 2.4
    np.testing.assert_allclose(plant.pose[0, :2], 0.0, atol=1e-3)


def test_mirror_symmetry():
    left, right = make(), make()
    speeds, angles = calculate_module_targets(np.array([2.0]), np.array([1.0]), np.array([0.5]), P.module_positions)
    speeds_m, angles_m = calculate_module_targets(np.array([2.0]), np.array([-1.0]), np.array([-0.5]), P.module_positions)
    for _ in range(50):
        left.step(angles, speeds, DT)
        right.step(angles_m, speeds_m, DT)
    np.testing.assert_allclose(right.pose[0], left.pose[0] * [1, -1, -1], atol=1e-6)


def test_batch_with_per_env_params_matches_single():
    mus = np.array([0.6, 0.9, 1.2])
    batch = RealisticPlant(3, per_env={"friction_coefficient": mus})
    batch.reset(np.zeros(3))
    run(batch, 0.3, 4.0, 0.5)
    for i, mu in enumerate(mus):
        single = make(friction_coefficient=mu)
        run(single, 0.3, 4.0, 0.5)
        np.testing.assert_allclose(batch.pose[i], single.pose[0], atol=1e-12)


def test_unknown_per_env_key_rejected():
    with pytest.raises(KeyError):
        RealisticPlant(1, per_env={"frictoin": 1.0})


def _odometry_vs_truth(plant, seconds, angle=0.0, speed=6.0):
    odo = Odometry(1)
    odo.reset(np.zeros(3), plant.drive_position, plant.module_angle)
    for _ in range(int(round(seconds / DT))):
        plant.step(np.full((1, 4), angle), np.full((1, 4), speed), DT)
        odo.update(plant.drive_position, plant.module_angle, plant.heading)
    return odo.pose[0, 0], plant.pose[0, 0]


def test_wheel_slip_makes_odometry_overestimate():
    est, true = _odometry_vs_truth(make(friction_coefficient=0.4), 0.5)
    assert est > true + 0.05


def test_worn_wheels_make_odometry_underestimate():
    # 實際輪徑比設定大 2% → 編碼器少算 2%
    # 高摩擦、重心 0：不打滑，只看輪徑誤差
    est, true = _odometry_vs_truth(make(wheel_radius_scale=1.02, friction_coefficient=1.5, cg_height_m=0.0), 1.0, speed=2.0)
    assert est == pytest.approx(true / 1.02, rel=0.005)
