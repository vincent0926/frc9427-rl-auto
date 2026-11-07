"""干擾模型：碰撞、推擠、牆與 HUB、地毯摩擦區塊、輪胎磨耗、壓到球、卡球。"""

import numpy as np
import pytest

from rl_auto.sim.chassis import RealisticPlant
from rl_auto.sim.disturbances import (BUMP_DURATION, HUB_CENTERS, HUB_HALF, ROBOT_HALF_LENGTH, Disturbances,
                                      contact_forces)
from rl_auto.sim.env import OBS_DIM_V2, SwerveAutoEnv
from rl_auto.sim.params import DEFAULT_PARAMS as P
from rl_auto.sim.trajectory import Trajectory

DT = P.cycle_time_s


def plant_at(x, y, heading=0.0, velocity=None, contact=True):
    pl = RealisticPlant(1, per_env={"cg_height_m": 0.0})
    if contact:
        pl.contact = contact_forces
    pl.reset(np.array([x, y, heading]))
    if velocity is not None:
        pl.set_motion(np.array([velocity]))
    return pl


def drive(pl, angle, speed, seconds):
    for _ in range(int(round(seconds / DT))):
        pl.step(np.full((1, 4), angle), np.full((1, 4), speed), DT)


def test_wall_stops_robot_at_bumper():
    pl = plant_at(1.0, 2.0, velocity=[-2.0, 0.0, 0.0])
    drive(pl, np.pi, 2.0, 2.0)  # 一直往左牆開
    assert pl.pose[0, 0] == pytest.approx(ROBOT_HALF_LENGTH, abs=0.02)
    assert abs(pl.velocity[0, 0]) < 0.05


def test_hub_pushes_robot_out():
    cx, cy = HUB_CENTERS[0]
    pl = plant_at(cx - HUB_HALF - ROBOT_HALF_LENGTH - 0.3, cy, velocity=[2.0, 0.0, 0.0])
    drive(pl, 0.0, 2.0, 1.5)  # 往 HUB 開
    assert pl.pose[0, 0] + ROBOT_HALF_LENGTH <= cx - HUB_HALF + 0.02


def test_no_contact_force_in_open_field():
    pl = plant_at(8.0, 4.0)
    fx, fy, tq = contact_forces(pl)
    assert fx[0] == fy[0] == tq[0] == 0.0


def make_dist(n=1, scale=1.0, seed=0):
    traj = Trajectory.load("Blue3-3")
    d = Disturbances(n, np.random.default_rng(seed), traj, P.module_positions)
    d.reset(np.ones(n, bool), scale, np.full(n, P.mass_kg), np.full(n, P.moi_kg_m2))
    return d


def test_scale_zero_means_no_disturbance():
    d = make_dist(64, scale=0.0)
    assert np.all(d.events == 0)
    pl = RealisticPlant(64)
    pl.reset(np.tile([5.0, 2.0, 0.0], (64, 1)))
    for t in np.linspace(0, 8, 50):
        d.update(np.full(64, t), pl)
        assert np.all(pl.mu_scale == 1.0) and np.all(pl.ext_force == 0.0)


def test_bump_changes_velocity_by_sampled_delta_v():
    d = make_dist()
    d.bump_time[0] = [1.0, np.inf]
    d.bump_force[0, 0] = [P.mass_kg * 0.8 / BUMP_DURATION, 0.0]  # 0.8 m/s
    d.bump_torque[0] = 0.0
    d.push_start[:] = d.drag_start[:] = np.inf
    d.ball_time[:] = np.inf
    d.zone_radius[:] = 0.0
    d.wear[:] = 1.0
    pl = RealisticPlant(1, per_env={"friction_coefficient": 0.0})  # 沒有輪胎力，只看外力
    pl.reset(np.array([8.0, 4.0, 0.0]))
    for k in range(60):
        d.update(np.array([k * DT]), pl)
        pl.step(np.zeros((1, 4)), np.zeros((1, 4)), DT)
    assert pl.velocity[0, 0] == pytest.approx(0.8, rel=0.05)


def test_ball_under_wheel_cuts_that_wheel_friction():
    d = make_dist()
    d.ball_time[0] = [1.0, np.inf, np.inf]
    d.ball_end[0] = [1.3, np.inf, np.inf]
    d.ball_module[0] = [2, 0, 0]
    d.zone_radius[:] = 0.0
    d.wear[:] = 1.0
    pl = RealisticPlant(1)
    pl.reset(np.array([8.0, 4.0, 0.0]))
    d.update(np.array([1.1]), pl)
    np.testing.assert_allclose(pl.mu_scale[0], [1.0, 1.0, 0.15, 1.0])
    d.update(np.array([1.4]), pl)
    np.testing.assert_allclose(pl.mu_scale[0], 1.0)


def test_friction_zone_only_affects_modules_inside():
    d = make_dist()
    d.wear[:] = 1.0
    d.ball_time[:] = np.inf
    d.zone_radius[0] = [0.3, 0.0, 0.0]
    d.zone_scale[0] = [0.5, 1.0, 1.0]
    pl = RealisticPlant(1)
    pl.reset(np.array([5.0, 2.0, 0.0]))
    fl = P.module_positions[0]
    d.zone_center[0, 0] = [5.0 + fl[0], 2.0 + fl[1]]  # 圓心放在左前輪
    d.update(np.array([0.0]), pl)
    np.testing.assert_allclose(pl.mu_scale[0], [0.5, 1.0, 1.0, 1.0])


def test_ball_jam_drag_opposes_motion():
    d = make_dist()
    d.drag_start[0], d.drag_end[0], d.drag_force[0] = 0.0, 5.0, 100.0
    d.bump_time[:] = np.inf
    d.push_start[:] = np.inf
    pl = RealisticPlant(1)
    pl.reset(np.array([5.0, 2.0, 0.0]))
    pl.set_motion(np.array([[0.0, 2.0, 0.0]]))
    d.update(np.array([1.0]), pl)
    np.testing.assert_allclose(pl.ext_force[0], [0.0, -100.0], atol=1e-9)


def test_full_scale_samples_events():
    d = make_dist(2000, scale=1.0)
    assert d.events.mean() > 2.0
    assert (d.bump_time < np.inf).any(axis=1).mean() > 0.4


def test_env_v2_observation_and_slip_indicator():
    env = SwerveAutoEnv("Blue3-2", n_envs=2, seed=0, randomize_scale=0.0, obs_version=2)
    obs = env.reset()
    assert obs.shape == (2, OBS_DIM_V2)
    for _ in range(40):
        obs, *_ = env.step(np.zeros((2, 4)))
    assert np.all(obs[:, -1] < 0.1)  # 正常行駛：四輪一致
    # 讓一輪在冰上 (摩擦 ×0.05) 硬加速 → 四輪不一致
    env.plant.mu_scale[0, 0] = 0.05
    env.disturbances.wear[0, 0] = 0.05
    for _ in range(10):
        obs, *_ = env.step(np.array([[1.0, 1.0, 0, 0], [0, 0, 0, 0]]))
    assert obs[0, -1] > obs[1, -1]


def test_env_with_disturbances_runs_and_reports_contacts():
    env = SwerveAutoEnv("Blue1-3", n_envs=64, seed=0, randomize_scale=1.0, disturbance_scale=1.0, obs_version=2)
    env.reset()
    finished = 0
    for _ in range(500):
        _, reward, _, _, info = env.step(np.zeros((64, 4)))
        assert np.all(np.isfinite(reward))
        finished += info["done"].sum()
    assert finished > 0
    assert "contact_steps" in info and "events" in info
