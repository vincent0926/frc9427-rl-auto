"""PPO 訓練程式的單元測試 (不跑完整訓練)。"""

import json

import numpy as np
import pytest
import torch

from rl_auto.sim.env import ACTION_DIM, OBS_DIM
from rl_auto.train.ppo import ActorCritic, RunningNorm, TrainConfig, export_policy, schedule


def test_running_norm_matches_batch_statistics():
    rng = np.random.default_rng(0)
    data = rng.normal(3.0, 2.0, (5000, 4))
    norm = RunningNorm(4)
    for chunk in np.array_split(data, 10):
        norm.update(chunk)
    np.testing.assert_allclose(norm.mean, data.mean(axis=0), atol=1e-3)
    np.testing.assert_allclose(norm.var, data.var(axis=0), rtol=1e-3)


def test_initial_policy_is_close_to_pathplanner():
    # 最後一層權重很小 → 一開始的動作平均 ≈ 0 (= PathPlanner)
    model = ActorCritic(OBS_DIM, ACTION_DIM, 64, -1.0)
    obs = torch.randn(256, OBS_DIM)
    assert model.dist(obs).mean.abs().max() < 0.1


def test_curriculum_schedule():
    cfg = TrainConfig()
    rand0, res0, stage0 = schedule(cfg, 0.0)
    assert (rand0, res0, stage0) == (pytest.approx(cfg.randomize_start), 0.0, 1)
    assert schedule(cfg, 0.29)[1] == 0.0  # 階段 1 只有時間縮放
    assert schedule(cfg, 0.5) == (pytest.approx(1.0), pytest.approx(0.5), 2)
    assert schedule(cfg, 0.9)[:2] == (1.0, 1.0)


def test_exported_policy_matches_torch(tmp_path):
    """機器人端 (Java) 會照匯出的 JSON 做推論；用純 numpy 照同樣算法算，必須跟 torch 一樣。"""
    torch.manual_seed(0)
    model = ActorCritic(OBS_DIM, ACTION_DIM, 64, -1.0)
    for p in model.actor.parameters():
        p.data.normal_(0, 0.3)
    norm = RunningNorm(OBS_DIM)
    norm.update(np.random.default_rng(1).normal(0, 2, (100, OBS_DIM)))
    out = tmp_path / "policy.json"
    export_policy(model, norm, out, TrainConfig())
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["obs_dim"] == OBS_DIM and data["action_dim"] == ACTION_DIM

    obs = np.random.default_rng(2).normal(0, 3, (50, OBS_DIM))
    x = np.clip((obs - np.array(data["obs_mean"])) / np.array(data["obs_std"]), -data["obs_clip"], data["obs_clip"])
    for i, layer in enumerate(data["layers"]):
        x = x @ np.array(layer["weight"]).T + np.array(layer["bias"])
        if i < len(data["layers"]) - 1:
            x = np.tanh(x)
    with torch.no_grad():
        expected = model.dist(torch.as_tensor(norm(obs), dtype=torch.float32)).mean.numpy()
    np.testing.assert_allclose(x, expected, atol=1e-4)
