"""第 3 步健全性檢查：機器人軟體 + 理想物理 (不打滑、無延遲) 跑 PathPlanner 路線應幾乎完美跟隨。

剩下的幾公分誤差來自機器人軟體本身 (模組 cos 縮放、20 ms 離散週期)，是 PathPlanner 基準的下限。
"""

import numpy as np
import pytest

from rl_auto.sim.chassis import IdealPlant
from rl_auto.sim.params import DEFAULT_PARAMS as P
from rl_auto.sim.robot_software import Odometry
from rl_auto.sim.rollout import PathFollower
from rl_auto.sim.trajectory import Trajectory


@pytest.mark.parametrize("name", Trajectory.available())
def test_ideal_plant_follows_every_path(name):
    traj = Trajectory.load(name)
    r = PathFollower(traj).run()
    assert not np.isnan(r.finish_time[0]), "沒有跑完"
    assert r.finish_time[0] <= traj.total_time + 0.1
    assert r.max_cross_track[0] < 0.06
    assert r.end_position_error[0] < 0.02
    assert np.degrees(r.end_heading_error[0]) < 0.5
    # 不打滑時里程計應該幾乎完美 (Java 改用週期中間朝向後)
    assert r.max_odometry_error[0] < 0.001


def test_batch_rollout_matches_single():
    traj = Trajectory.load("Blue3-2")
    single = PathFollower(traj, n_envs=1).run()
    batch = PathFollower(traj, n_envs=4).run()
    np.testing.assert_allclose(batch.finish_time, single.finish_time[0])
    np.testing.assert_allclose(batch.max_cross_track, single.max_cross_track[0], atol=1e-12)


def _drive(plant, odo, angle, speed, cycles):
    for _ in range(cycles):
        plant.step(np.full((1, 4), angle) if np.ndim(angle) == 0 else angle, np.full((1, 4), speed) if np.ndim(speed) == 0 else speed, P.cycle_time_s)
        odo.update(plant.drive_position, plant.module_angle, plant.heading)


def test_ideal_plant_straight_line():
    plant = IdealPlant(1)
    plant.reset(np.zeros(3))
    odo = Odometry(1)
    odo.reset(np.zeros(3), plant.drive_position, plant.module_angle)
    _drive(plant, odo, 0.0, 2.0, 50)  # 2 m/s × 1 s
    np.testing.assert_allclose(plant.pose[0], [2.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(odo.pose[0], plant.pose[0], atol=1e-9)


def test_ideal_plant_spin_in_place():
    plant = IdealPlant(1)
    plant.reset(np.zeros(3))
    pos = P.module_positions
    radius = np.hypot(pos[:, 0], pos[:, 1])
    angle = np.arctan2(pos[:, 0], -pos[:, 1])[None, :]  # 逆時針切線方向
    omega = 1.0
    plant.module_angle = angle.copy()  # 模組先轉好，避免第一個週期轉向
    for _ in range(50):
        plant.step(angle, (omega * radius)[None, :], P.cycle_time_s)
    np.testing.assert_allclose(plant.pose[0, :2], 0.0, atol=1e-9)
    assert plant.pose[0, 2] == pytest.approx(1.0)


def test_odometry_drives_and_turns_without_drift():
    # 邊直走邊原地轉：真實位置與里程計要一致 (舊 Java 用週期結束朝向時會偏)
    plant = IdealPlant(1)
    plant.reset(np.zeros(3))
    odo = Odometry(1)
    odo.reset(np.zeros(3), plant.drive_position, plant.module_angle)
    pos = P.module_positions
    from rl_auto.sim.robot_software import calculate_module_targets, field_to_robot

    for _ in range(100):
        rvx, rvy = field_to_robot(2.0, 0.0, plant.heading)
        speed, angle = calculate_module_targets(rvx, rvy, np.array([2.0]), pos)
        plant.step(angle, speed, P.cycle_time_s)
        odo.update(plant.drive_position, plant.module_angle, plant.heading)
    np.testing.assert_allclose(odo.pose[0, :2], plant.pose[0, :2], atol=2e-3)
