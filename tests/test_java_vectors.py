"""對照 Java：data/robot-export/vectors/ 由機器人專案的 RlExportTest 匯出 (Java float 運算)。

Python 用 float64，所以容許誤差取 1e-4 等級。改機器人程式後要重新匯出並跑這組測試。
"""

from pathlib import Path

import numpy as np
import pytest

from rl_auto.sim.params import DEFAULT_PARAMS as P
from rl_auto.sim.robot_software import arc_displacement, calculate_module_targets, desaturate, optimize_module

VECTORS = Path(__file__).resolve().parents[1] / "data" / "robot-export" / "vectors"


def load(name):
    return np.genfromtxt(VECTORS / f"{name}.csv", delimiter=",", names=True)


def test_module_targets_and_desaturate_match_java():
    d = load("module_targets")
    speeds, angles = calculate_module_targets(d["vx"], d["vy"], d["omega"], P.module_positions)
    speeds, _ = desaturate(speeds, P.max_module_speed_mps)
    java_speeds = np.stack([d[f"s{i}"] for i in range(4)], axis=1)
    java_angles = np.stack([d[f"a{i}"] for i in range(4)], axis=1)
    np.testing.assert_allclose(speeds, java_speeds, atol=1e-4)
    np.testing.assert_allclose(angles, java_angles, atol=1e-4)


def test_arc_displacement_matches_java():
    d = load("arc_displacement")
    dx, dy = arc_displacement(d["prev_angle"], d["angle"], d["distance"])
    np.testing.assert_allclose(dx, d["dx"], atol=1e-4)
    np.testing.assert_allclose(dy, d["dy"], atol=1e-4)


def test_module_optimize_matches_java():
    d = load("module_optimize")
    angle, speed = optimize_module(d["wanted_angle"], d["wanted_speed"], d["current_angle"])
    np.testing.assert_allclose(angle, d["target_angle"], atol=1e-4)
    np.testing.assert_allclose(speed, d["target_speed"], atol=1e-4)


def test_vectors_exist():
    for name in ("module_targets", "arc_displacement", "module_optimize"):
        assert len(load(name)) == 500, name
