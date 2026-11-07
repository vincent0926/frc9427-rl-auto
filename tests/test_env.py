"""向量化訓練環境、領域隨機化的測試。"""

import numpy as np
import pytest

from rl_auto.sim.env import ACTION_DIM, OBS_DIM, SwerveAutoEnv, evaluate_policy
from rl_auto.sim.params import DEFAULT_PARAMS as P
from rl_auto.sim.randomization import Randomizer, physics_ranges
from rl_auto.sim.sensors import SensorParams


def zero(obs):
    return np.zeros((obs.shape[0], ACTION_DIM))


def test_shapes_and_finite():
    env = SwerveAutoEnv("Blue3-2", n_envs=8, seed=0)
    obs = env.reset()
    assert obs.shape == (8, OBS_DIM)
    for _ in range(20):
        obs, reward, terminated, truncated, info = env.step(np.random.default_rng(0).uniform(-1, 1, (8, ACTION_DIM)))
        assert obs.shape == (8, OBS_DIM) and reward.shape == (8,)
        assert np.all(np.isfinite(obs)) and np.all(np.isfinite(reward))


def test_zero_action_without_randomization_matches_pathplanner():
    env = SwerveAutoEnv("Blue3-2", n_envs=4, seed=0, randomize_scale=0.0)
    r = evaluate_policy(env, zero, 4)
    assert r["success_rate"] == 1.0
    assert r["mean_time_success"] == pytest.approx(env.traj.total_time, abs=0.1)


def test_auto_reset_after_done():
    env = SwerveAutoEnv("Blue3-1", n_envs=2, seed=0, randomize_scale=0.0)
    env.reset()
    for _ in range(400):
        _, _, _, _, info = env.step(zero(np.zeros((2, OBS_DIM))))
        if info["done"].any():
            done = info["done"]
            assert np.all(env.follower.tau[done] == 0.0)
            assert np.all(env.follower.time[done] == 0.0)
            assert "final_observation" in info
            return
    pytest.fail("沒有任何 episode 結束")


def test_same_seed_same_result():
    runs = []
    for _ in range(2):
        env = SwerveAutoEnv("Blue3-2", n_envs=4, seed=5)
        env.reset()
        total = np.zeros(4)
        for _ in range(50):
            _, reward, _, _, _ = env.step(np.full((4, ACTION_DIM), 0.2))
            total += reward
        runs.append(total)
    np.testing.assert_array_equal(runs[0], runs[1])


def test_residual_scale_zero_ignores_corrections():
    results = []
    for residual_action in (0.0, 1.0):
        env = SwerveAutoEnv("Blue3-2", n_envs=2, seed=1, randomize_scale=0.0, residual_scale=0.0)
        env.reset()
        action = np.zeros((2, ACTION_DIM))
        action[:, 1:] = residual_action
        for _ in range(30):
            env.step(action)
        results.append(env.plant.pose.copy())
    np.testing.assert_allclose(results[0], results[1])


def test_time_scaling_speeds_up_reference():
    env = SwerveAutoEnv("Blue3-2", n_envs=1, seed=0, randomize_scale=0.0)
    env.reset()
    for _ in range(10):
        env.step(np.array([[1.0, 0, 0, 0]]))  # s = 1.5
    assert env.follower.tau[0] == pytest.approx(10 * 0.02 * 1.5)


def test_residual_alone_cannot_push_robot_far_off_path():
    # 安全性質：殘差上限 0.5 m/s，PathPlanner P (kP 5) 會拉回，穩態偏移約 0.5 / 5 = 0.1 m
    env = SwerveAutoEnv("Blue3-3", n_envs=1, seed=0, randomize_scale=0.0, residual_scale=1.0)
    env.reset()
    for _ in range(200):
        _, _, terminated, _, info = env.step(np.array([[0.0, 0.0, 1.0, 0.0]]))
        assert not info["failure"][0]
    assert env.ep_max_ct[0] < 0.2


def test_cross_track_failure_terminates_with_penalty():
    env = SwerveAutoEnv("Blue3-3", n_envs=1, seed=0, randomize_scale=0.0, residual_scale=1.0,
                        cross_track_limit_m=0.05)
    env.reset()
    for _ in range(200):
        _, reward, terminated, _, info = env.step(np.array([[0.0, 0.0, 1.0, 0.0]]))  # 一直往側邊推
        if terminated[0]:
            assert info["failure"][0]
            assert reward[0] < -5
            return
    pytest.fail("偏線沒有終止")


def test_randomizer_scale_zero_gives_defaults():
    rnd = Randomizer(np.random.default_rng(0), scale=0.0)
    phys = rnd.sample_physics(5)
    np.testing.assert_allclose(phys["friction_coefficient"], P.friction_coefficient)
    np.testing.assert_allclose(phys["wheel_radius_scale"], 1.0)
    sens = rnd.sample_sensors(5, SensorParams())
    np.testing.assert_array_equal(sens["command_delay_cycles"], 1)
    np.testing.assert_allclose(rnd.sample_start_offset(5), 0.0)


def test_randomizer_full_scale_covers_ranges():
    rnd = Randomizer(np.random.default_rng(0), scale=1.0)
    phys = rnd.sample_physics(5000)
    for key, (lo, hi) in physics_ranges().items():
        assert phys[key].min() >= lo - 1e-9 and phys[key].max() <= hi + 1e-9, key
        assert phys[key].max() - phys[key].min() > 0.8 * (hi - lo), key
    delays = rnd.sample_sensors(5000, SensorParams())["command_delay_cycles"]
    assert set(np.unique(delays)) == {0, 1, 2, 3}


def test_randomized_params_reach_plant():
    env = SwerveAutoEnv("Blue3-2", n_envs=64, seed=0, randomize_scale=1.0)
    env.reset()
    assert env.plant.mu.std() > 0.1
    assert env.follower.camera.latency.std() > 0.005
