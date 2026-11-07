"""感測模型與視覺融合 (Swerve.addVisionMeasurement / PoseHistory 移植) 的測試。"""

import numpy as np
import pytest

from rl_auto.sim.chassis import RealisticPlant
from rl_auto.sim.params import DEFAULT_PARAMS as P
from rl_auto.sim.robot_software import Odometry
from rl_auto.sim.rollout import PathFollower
from rl_auto.sim.sensors import (CommandDelay, Gyro, PoseHistory, SensorParams, VISION_BLEND_GAIN,
                                 VisionCamera, VisionFusion)
from rl_auto.sim.trajectory import Trajectory


def odometry_at(x, y, n=1):
    odo = Odometry(n)
    odo.reset(np.tile([x, y, 0.0], (n, 1)), np.zeros((n, 4)), np.zeros((n, 4)))
    return odo


# ---- PoseHistory -----------------------------------------------------------

def test_history_interpolates_and_reports_range():
    h = PoseHistory(1, capacity=5)
    for k in range(3):
        h.add(np.array([0.1 * k]), np.array([float(k)]), np.array([2.0 * k]))
    x, y, status = h.sample(np.array([0.15]))
    assert status[0] == 0
    assert x[0] == pytest.approx(1.5)
    assert y[0] == pytest.approx(3.0)
    assert h.sample(np.array([-1.0]))[2][0] == -1  # 比最舊還早
    assert h.sample(np.array([1.0]))[2][0] == 1  # 比最新還新


def test_history_ring_buffer_drops_oldest():
    h = PoseHistory(1, capacity=3)
    for k in range(5):
        h.add(np.array([float(k)]), np.array([float(k)]), np.array([0.0]))
    assert h.sample(np.array([1.0]))[2][0] == -1  # 0、1 已被蓋掉
    assert h.sample(np.array([3.5]))[0][0] == pytest.approx(3.5)


# ---- VisionFusion (同 Java addVisionMeasurement) -----------------------------

def test_first_measurement_aligns_then_blends():
    odo = odometry_at(1.0, 1.0)
    fusion = VisionFusion(1)
    fusion.record(np.array([0.0]), odo)
    applied = fusion.add_measurement(odo, np.array([1.5]), np.array([1.0]), np.array([0.0]), np.array([True]))
    assert applied[0]
    assert odo.pose[0, 0] == pytest.approx(1.5)  # 第一筆直接對齊
    fusion.add_measurement(odo, np.array([2.5]), np.array([1.0]), np.array([0.0]), np.array([True]))
    # 第二筆往視覺靠近 10%；參考位置是拍攝當下 (已被第一筆平移過的紀錄 = 1.5)
    assert odo.pose[0, 0] == pytest.approx(1.5 + VISION_BLEND_GAIN * 1.0)


def test_jump_over_two_meters_rejected():
    odo = odometry_at(1.0, 1.0)
    fusion = VisionFusion(1)
    applied = fusion.add_measurement(odo, np.array([4.0]), np.array([1.0]), np.array([0.0]), np.array([True]))
    assert not applied[0]
    assert odo.pose[0, 0] == pytest.approx(1.0)


def test_zero_or_nan_measurement_ignored():
    odo = odometry_at(1.0, 1.0)
    fusion = VisionFusion(1)
    for vx, vy in [(0.0, 0.0), (np.nan, 1.0)]:
        applied = fusion.add_measurement(odo, np.array([vx]), np.array([vy]), np.array([0.0]), np.array([True]))
        assert not applied[0]


def test_latency_compensation_uses_pose_at_capture_time():
    # 車子在動，里程計正確；視覺是 0.1 秒前拍的真實位置 → 不應該修正
    odo = odometry_at(0.0, 0.0)
    fusion = VisionFusion(1)
    fusion.has_fix[:] = True
    for k in range(11):
        odo.pose[0, 0] = 0.4 * k * 0.02 / 0.02 * 0.02  # 每週期 0.008 m
        fusion.record(np.array([k * 0.02]), odo)
    capture = 0.2 - 0.1
    true_at_capture = 0.008 * 5
    before = odo.pose[0, 0]
    fusion.add_measurement(odo, np.array([true_at_capture]), np.array([0.0]), np.array([capture]), np.array([True]))
    assert odo.pose[0, 0] == pytest.approx(before, abs=1e-9)


def test_measurement_older_than_history_dropped():
    odo = odometry_at(0.0, 0.0)
    fusion = VisionFusion(1)
    fusion.record(np.array([1.0]), odo)
    applied = fusion.add_measurement(odo, np.array([0.5]), np.array([0.0]), np.array([0.5]), np.array([True]))
    assert not applied[0]


# ---- 感測器 ------------------------------------------------------------------

def test_camera_rate_latency_and_dropout():
    rng = np.random.default_rng(0)
    cam = VisionCamera(1, SensorParams(vision_rate_hz=10.0, vision_latency_s=0.05, vision_dropout=0.0,
                                       vision_sigma_m=0.0), rng)
    cam.reset()
    arrivals = []
    pose = np.array([[1.0, 2.0, 0.0]])
    for k in range(100):  # 2 秒
        t = np.array([k * 0.02])
        x, y, capture, arrived = cam.update(t, pose)
        if arrived[0]:
            arrivals.append((t[0], capture[0], x[0], y[0]))
    assert 19 <= len(arrivals) <= 21  # 10 Hz
    for t, capture, x, y in arrivals:
        assert t - capture >= 0.05 - 1e-9
        assert (x, y) == (1.0, 2.0)


def test_camera_dropout_reduces_frames():
    rng = np.random.default_rng(0)
    cam = VisionCamera(2000, SensorParams(vision_rate_hz=10.0, vision_dropout=0.5), rng)
    cam.reset()
    total = 0
    for k in range(50):
        total += cam.update(np.full(2000, k * 0.02), np.zeros((2000, 3)))[3].sum()
    # 1 秒拍 10 張、一半掉幀 → 約 5 張；最後一張可能還在途中，所以略少
    assert 4.3 < total / 2000 < 5.0


def test_gyro_without_drift_is_exact():
    gyro = Gyro(3, SensorParams(gyro_drift_rad_per_sqrt_s=0.0), np.random.default_rng(0))
    np.testing.assert_allclose(gyro.read(np.array([0.1, -1.0, 3.0]), 0.02), [0.1, -1.0, 3.0])


def test_command_delay_shifts_by_cycles():
    delay = CommandDelay(3, SensorParams(command_delay_cycles=np.array([0, 1, 2])))
    delay.reset(np.zeros((3, 4)))
    outputs = []
    for k in range(1, 5):
        _, speed = delay.push(np.zeros((3, 4)), np.full((3, 4), float(k)))
        outputs.append(speed[:, 0])
    outputs = np.array(outputs)  # (週期, 環境)
    np.testing.assert_allclose(outputs[:, 0], [1, 2, 3, 4])
    np.testing.assert_allclose(outputs[:, 1], [0, 1, 2, 3])
    np.testing.assert_allclose(outputs[:, 2], [0, 0, 1, 2])


# ---- 整合 --------------------------------------------------------------------

def test_vision_corrects_odometry_drift_from_slip():
    traj = Trajectory.load("Blue3-3")
    plant = lambda: RealisticPlant(1, per_env={"friction_coefficient": 0.6})
    blind = PathFollower(traj, plant=plant(), sensors=SensorParams(vision_enabled=False), seed=3).run()
    seeing = PathFollower(traj, plant=plant(), sensors=SensorParams(), seed=3).run()
    assert seeing.end_position_error[0] < blind.end_position_error[0] * 0.5


def test_rollout_with_sensors_is_deterministic():
    traj = Trajectory.load("Blue3-2")
    runs = [PathFollower(traj, n_envs=2, plant=RealisticPlant(2), sensors=SensorParams(), seed=7).run()
            for _ in range(2)]
    np.testing.assert_array_equal(runs[0].max_cross_track, runs[1].max_cross_track)


def test_ideal_sensors_match_no_sensor_path():
    # 無漂移、無延遲、無視覺 → 跟 sensors=None 一樣
    traj = Trajectory.load("Blue1-2")
    off = SensorParams(vision_enabled=False, gyro_drift_rad_per_sqrt_s=0.0, command_delay_cycles=0)
    a = PathFollower(traj, sensors=off).run()
    b = PathFollower(traj).run()
    np.testing.assert_allclose(a.max_cross_track, b.max_cross_track, atol=1e-12)
