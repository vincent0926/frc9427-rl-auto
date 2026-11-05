"""robot_software 移植的單元測試。預期值依 Java 原始碼 (SwerveMath / SwerveModule / Swerve) 的行為推得。"""

import numpy as np
import pytest

from rl_auto.sim.params import DEFAULT_PARAMS as P
from rl_auto.sim.robot_software import (
    Odometry,
    SwerveSoftware,
    arc_displacement,
    calculate_module_targets,
    desaturate,
    field_to_robot,
    optimize_module,
    pp_holonomic_controller,
    robot_to_field,
    wrap_rad,
)


def test_max_module_speed_matches_constants():
    # Constants.Swerve.MaxModuleSpeedMps = (11 − 0.18) × 0.85 / DriveVoltsPerMps ≈ 4.64
    assert P.drive_volts_per_mps == pytest.approx(1.981, abs=1e-3)
    assert P.max_module_speed_mps == pytest.approx(4.64, abs=0.01)
    assert P.free_speed_mps == pytest.approx(6.06, abs=0.01)


@pytest.mark.parametrize(
    "value, expected",
    [(0.0, 0.0), (np.pi, -np.pi), (-np.pi, -np.pi), (3 * np.pi / 2, -np.pi / 2), (-3 * np.pi / 2, np.pi / 2)],
)
def test_wrap_rad(value, expected):
    assert wrap_rad(value) == pytest.approx(expected)


def test_field_robot_roundtrip():
    rng = np.random.default_rng(0)
    x, y, h = rng.normal(size=(3, 100))
    rx, ry = field_to_robot(x, y, h)
    fx, fy = robot_to_field(rx, ry, h)
    np.testing.assert_allclose([fx, fy], [x, y], atol=1e-12)


def test_field_to_robot_heading_90deg():
    # 朝向 +90°：場地 +x 方向在機器人座標是 −y (右邊)
    rx, ry = field_to_robot(1.0, 0.0, np.pi / 2)
    assert rx == pytest.approx(0.0, abs=1e-12)
    assert ry == pytest.approx(-1.0)


def test_module_targets_pure_translation():
    speeds, angles = calculate_module_targets(np.array([3.0]), np.array([4.0]), np.array([0.0]), P.module_positions)
    np.testing.assert_allclose(speeds, 5.0)
    np.testing.assert_allclose(angles, np.arctan2(4.0, 3.0))


def test_module_targets_pure_rotation():
    speeds, angles = calculate_module_targets(np.array([0.0]), np.array([0.0]), np.array([2.0]), P.module_positions)
    radius = np.hypot(P.wheel_base_m / 2, P.track_width_m / 2)
    np.testing.assert_allclose(speeds, 2.0 * radius)
    # 逆時針旋轉：FL 模組 (前左) 速度方向朝後左 → 角度 = atan2(x, -y)
    fl = P.module_positions[0]
    assert angles[0, 0] == pytest.approx(np.arctan2(fl[0], -fl[1]))


def test_desaturate_keeps_ratio():
    speeds = np.array([[6.0, 3.0, -2.0, 1.0]])
    out, scale = desaturate(speeds, 4.0)
    np.testing.assert_allclose(out, speeds * (4.0 / 6.0))
    assert scale[0] == pytest.approx(4.0 / 6.0)


def test_desaturate_untouched_below_max():
    speeds = np.array([[1.0, -2.0, 3.0, 0.5]])
    out, scale = desaturate(speeds, 4.0)
    np.testing.assert_allclose(out, speeds)
    assert scale[0] == 1.0


def test_optimize_flips_when_over_90deg():
    angle, speed = optimize_module(np.pi, 2.0, 0.0)
    # 要轉 180° → 不轉，速度反向
    assert speed == pytest.approx(-2.0)
    assert wrap_rad(angle) == pytest.approx(0.0, abs=1e-12)


def test_optimize_cos_scaling():
    angle, speed = optimize_module(np.pi / 3, 2.0, 0.0)
    assert angle == pytest.approx(np.pi / 3)
    assert speed == pytest.approx(2.0 * 0.5)


def test_arc_displacement_straight():
    dx, dy = arc_displacement(0.3, 0.3, 2.0)
    assert dx == pytest.approx(2.0 * np.cos(0.3))
    assert dy == pytest.approx(2.0 * np.sin(0.3))


def test_arc_displacement_quarter_circle():
    # 從 0 轉到 90°，走四分之一圓弧長 → 位移是半徑 R 的 (R, R)
    radius = 1.0
    dx, dy = arc_displacement(0.0, np.pi / 2, np.pi / 2 * radius)
    assert dx == pytest.approx(radius)
    assert dy == pytest.approx(radius)


def test_pp_controller_feedforward_plus_p():
    vx, vy, w = pp_holonomic_controller(1.0, 2.0, 0.5, 3.0, 0.0, 1.0, 0.9, 2.0, 0.4, 5.0, 5.0)
    assert vx == pytest.approx(3.0 + 5.0 * 0.1)
    assert vy == pytest.approx(0.0)
    assert w == pytest.approx(1.0 + 5.0 * 0.1)


def test_follow_path_caps_at_max_module_speed():
    sw = SwerveSoftware(1)
    angle, speed = sw.follow_path(np.array([5.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]), np.zeros((1, 4)))
    np.testing.assert_allclose(speed, P.max_module_speed_mps)
    np.testing.assert_allclose(angle, 0.0)


def test_follow_path_idle_holds_current_angle():
    # SwerveModule.stop()：targetAngleRad = 目前實際角度
    sw = SwerveSoftware(1)
    sw.follow_path(np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([0.0]), np.zeros((1, 4)))
    current = np.array([[0.3, 1.2, 1.5, 1.6]])
    angle, speed = sw.follow_path(np.zeros(1), np.zeros(1), np.zeros(1), np.zeros(1), current)
    np.testing.assert_allclose(speed, 0.0)
    np.testing.assert_allclose(angle, current)


def test_follow_path_slow_module_keeps_previous_target():
    # 模組速度 < ModuleHoldSpeedMps → setTarget(上一個目標角度, 0)
    sw = SwerveSoftware(1)
    sw.follow_path(np.array([0.0]), np.array([1.0]), np.array([0.0]), np.array([0.0]), np.zeros((1, 4)))
    previous = sw.target_angle.copy()
    angle, speed = sw.follow_path(np.array([0.01]), np.zeros(1), np.zeros(1), np.zeros(1), previous)
    np.testing.assert_allclose(speed, 0.0)
    np.testing.assert_allclose(angle, previous)


def test_follow_path_is_vectorized():
    rng = np.random.default_rng(1)
    n = 64
    vx, vy, w, h = rng.normal(size=(4, n)) * 2
    cur = rng.uniform(-np.pi, np.pi, size=(n, 4))
    batch_angle, batch_speed = SwerveSoftware(n).follow_path(vx, vy, w, h, cur)
    for i in range(n):
        a, s = SwerveSoftware(1).follow_path(vx[i:i + 1], vy[i:i + 1], w[i:i + 1], h[i:i + 1], cur[i:i + 1])
        np.testing.assert_allclose(batch_angle[i], a[0])
        np.testing.assert_allclose(batch_speed[i], s[0])


def test_odometry_straight_line():
    odo = Odometry(1)
    odo.reset(np.zeros(3), np.zeros((1, 4)), np.zeros((1, 4)))
    for k in range(1, 51):
        odo.update(np.full((1, 4), 0.02 * k), np.zeros((1, 4)), np.zeros(1))
    np.testing.assert_allclose(odo.pose[0], [1.0, 0.0, 0.0], atol=1e-9)


def test_odometry_follows_heading():
    # 朝向 90°、模組朝機器人前方 → 場地上往 +y 走
    odo = Odometry(1)
    odo.reset(np.array([0.0, 0.0, np.pi / 2]), np.zeros((1, 4)), np.zeros((1, 4)))
    odo.update(np.full((1, 4), 1.0), np.zeros((1, 4)), np.array([np.pi / 2]))
    np.testing.assert_allclose(odo.pose[0], [0.0, 1.0, np.pi / 2], atol=1e-9)
