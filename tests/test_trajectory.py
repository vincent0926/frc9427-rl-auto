import numpy as np
import pytest

from rl_auto.sim.trajectory import Trajectory


def straight(length=4.0, speed=2.0, dt=0.005):
    t = np.arange(0.0, length / speed + dt / 2, dt)
    x = speed * t
    zeros = np.zeros_like(t)
    return Trajectory("straight", t, x, zeros, zeros, np.full_like(t, speed), zeros, zeros)


@pytest.mark.parametrize("name", Trajectory.available())
def test_exported_paths_are_sane(name):
    traj = Trajectory.load(name)
    assert traj.total_time > 0.5
    assert np.all(np.diff(traj.t) > 0)
    # 平均速度不可能超過模組上限 (≈4.64 m/s)：擋住 PathPlanner 產生的壞軌跡
    assert traj.length / traj.total_time < 4.7
    # 速度積分 (梯形法) ≈ 位置變化。PathPlanner 狀態間是內插，誤差隨路長累積：容許路長的 2%
    # (有些路線繞一圈回起點，位移接近 0，不能用相對誤差)
    dt = np.diff(traj.t)
    for v, p in ((traj.vx, traj.x), (traj.vy, traj.y)):
        integral = np.sum(0.5 * (v[1:] + v[:-1]) * dt)
        assert abs(integral - (p[-1] - p[0])) < 0.02 * traj.length


def test_expected_paths_exported():
    assert {"Blue1-1", "Blue1-2", "Blue1-3", "Blue3-1", "Blue3-2", "Blue3-3"} <= set(Trajectory.available())


def test_sample_interpolates_and_clamps():
    traj = straight()
    s = traj.sample(np.array([-1.0, 0.5, 99.0]))
    np.testing.assert_allclose(s.x, [0.0, 1.0, 4.0])
    np.testing.assert_allclose(s.vx, 2.0)


def test_sample_time_scaling_scales_velocity_only():
    traj = straight()
    s = traj.sample(np.array([0.5, 0.5]), scale=np.array([1.0, 1.5]))
    np.testing.assert_allclose(s.x, [1.0, 1.0])
    np.testing.assert_allclose(s.vx, [2.0, 3.0])


def test_heading_interpolates_across_pi():
    t = np.array([0.0, 1.0])
    zeros = np.zeros(2)
    traj = Trajectory("wrap", t, zeros, zeros, np.array([3.0, -3.0]), zeros, zeros, zeros)
    mid = traj.sample(np.array([0.5])).heading[0]
    # 3.0 → -3.0 走短路經過 π，中點在 ±π 附近，不是 0
    assert abs(abs(mid) - np.pi) < 1e-9


def test_cross_track_and_progress():
    traj = straight()
    dist, progress = traj.cross_track(np.array([1.0, 2.5, 5.0]), np.array([0.0, -0.3, 0.0]))
    np.testing.assert_allclose(dist, [0.0, 0.3, 1.0], atol=1e-12)
    np.testing.assert_allclose(progress, [1.0, 2.5, 4.0], atol=1e-12)
