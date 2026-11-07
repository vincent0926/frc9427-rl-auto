"""PPO 訓練 (殘差策略)。

- 策略：小型 MLP (2 × 64，tanh)，輸出高斯分布的平均值；log_std 是可學參數。規格書要求在 roboRIO 上用純 Java
  推論，所以網路刻意做小，並把觀測正規化參數一起匯出 (export_policy)。
- 課程學習：階段 1 只開時間縮放 (residual_scale = 0)、隨機化範圍由窄變寬；階段 2 逐步解鎖速度修正。
- 每隔幾次更新用固定種子評估 (確定性策略 = 平均值)，跟 PathPlanner 基準 B1 比；結果寫進 runs/<名稱>/
  給監控網頁 (dashboard/) 讀。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..sim.env import ACTION_DIM, SwerveAutoEnv, evaluate_policy, obs_dim

RUNS_DIR = Path(__file__).resolve().parents[2] / "runs"


def atomic_write(path: Path, text: str, required: bool = False):
    """先寫暫存檔再取代。Windows 上若監控網頁剛好在讀這個檔，取代會被拒絕 → 重試幾次；
    監控用的檔案 (required=False) 最後還是失敗就跳過，不能讓訓練因此中斷。"""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    for attempt in range(20):
        try:
            tmp.replace(path)
            return True
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    if required:
        raise PermissionError(f"無法寫入 {path}")
    print(f"警告：{path.name} 被占用，這次跳過", flush=True)
    return False


@dataclass
class TrainConfig:
    path: str = "Blue3-3"
    total_steps: int = 30_000_000
    n_envs: int = 512
    n_steps: int = 64  # 每次更新每個環境跑幾步
    epochs: int = 5
    minibatch: int = 4096
    lr: float = 3e-4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    max_grad_norm: float = 0.5
    init_log_std: float = -1.0
    hidden: int = 64
    # 課程學習 (以總步數比例表示)
    stage1_fraction: float = 0.3  # 前 30%：只有時間縮放
    randomize_start: float = 0.3  # 隨機化範圍從 30% 開始
    randomize_full_at: float = 0.5  # 到 50% 時用完整範圍
    residual_full_at: float = 0.7  # 速度修正在 30%→70% 之間從 0 開到 1
    disturbance_start: float = 0.15  # 干擾 (碰撞、推擠、摩擦不均、壓球、卡球) 從 15% 開始出現
    disturbance_full_at: float = 0.6  # 到 60% 時完整強度
    eval_disturbance: float = 1.0  # 評估時的干擾強度 (固定，曲線才能比較)
    obs_version: int = 2  # 1 = v1 模型的觀測；2 = 加上加速度計與打滑指標
    eval_every: int = 10  # 每幾次更新評估一次
    eval_episodes: int = 256
    seed: int = 0
    threads: int = 8


class RunningNorm:
    """觀測正規化：累積平均與變異數 (Welford 合併)。訓練後凍結並匯出給 Java。"""

    def __init__(self, dim: int):
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        b_mean, b_var, b_count = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = b_mean - self.mean
        total = self.count + b_count
        self.mean = self.mean + delta * b_count / total
        m2 = self.var * self.count + b_var * b_count + delta ** 2 * self.count * b_count / total
        self.var = m2 / total
        self.count = total

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int, init_log_std: float):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                                   nn.Linear(hidden, act_dim))
        self.critic = nn.Sequential(nn.Linear(obs_dim, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(),
                                    nn.Linear(256, 1))
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))
        # 最後一層輸出接近 0：一開始的策略 ≈ PathPlanner (動作 0)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        nn.init.zeros_(self.actor[-1].bias)

    def dist(self, obs):
        mean = self.actor(obs)
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)


def disturbance_schedule(cfg: TrainConfig, progress: float) -> float:
    span = max(cfg.disturbance_full_at - cfg.disturbance_start, 1e-9)
    return float(np.clip((progress - cfg.disturbance_start) / span, 0.0, 1.0))


def schedule(cfg: TrainConfig, progress: float) -> tuple[float, float, int]:
    """回傳 (隨機化範圍, 殘差比例, 階段)。progress = 已訓練步數 / 總步數。"""
    if progress < cfg.randomize_full_at:
        rand = cfg.randomize_start + (1.0 - cfg.randomize_start) * progress / cfg.randomize_full_at
    else:
        rand = 1.0
    if progress < cfg.stage1_fraction:
        return rand, 0.0, 1
    residual = min((progress - cfg.stage1_fraction) / (cfg.residual_full_at - cfg.stage1_fraction), 1.0)
    return rand, residual, 2


class Trainer:
    def __init__(self, cfg: TrainConfig, run_name: str | None = None):
        self.cfg = cfg
        torch.manual_seed(cfg.seed)
        torch.set_num_threads(cfg.threads)
        self.run_name = run_name or time.strftime(f"{cfg.path}-%Y%m%d-%H%M%S")
        self.dir = RUNS_DIR / self.run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

        rand, residual, _ = schedule(cfg, 0.0)
        self.obs_dim = obs_dim(cfg.obs_version)
        self.env = SwerveAutoEnv(cfg.path, cfg.n_envs, seed=cfg.seed, randomize_scale=rand, residual_scale=residual,
                                 disturbance_scale=disturbance_schedule(cfg, 0.0), obs_version=cfg.obs_version)
        self.model = ActorCritic(self.obs_dim, ACTION_DIM, cfg.hidden, cfg.init_log_std)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, eps=1e-5)
        self.norm = RunningNorm(self.obs_dim)
        self.steps = 0
        self.update = 0
        self.best_score = -math.inf
        self.metrics_file = open(self.dir / "metrics.jsonl", "a", encoding="utf-8")
        self.recent = {"return": [], "time": [], "success": [], "max_ct": []}

    # ---- 紀錄 ---------------------------------------------------------------
    def log(self, record: dict):
        record["wall_time"] = time.time()
        self.metrics_file.write(json.dumps(record) + "\n")
        self.metrics_file.flush()

    def status(self, state: str, **extra):
        data = {"state": state, "run": self.run_name, "steps": self.steps, "total_steps": self.cfg.total_steps,
                "update": self.update, "path": self.cfg.path, "updated": time.time(), **extra}
        atomic_write(self.dir / "status.json", json.dumps(data))

    # ---- 策略 ---------------------------------------------------------------
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            o = torch.as_tensor(self.norm(obs), dtype=torch.float32)
            d = self.model.dist(o)
            a = d.mean if deterministic else d.sample()
        return a.numpy()

    # ---- 主迴圈 -------------------------------------------------------------
    def train(self):
        cfg = self.cfg
        n, t_len = cfg.n_envs, cfg.n_steps
        start = time.time()
        self.status("starting")
        self.baseline = self.evaluate_baseline()
        obs = self.env.reset()
        self.norm.update(obs)

        while self.steps < cfg.total_steps:
            rand, residual, stage = schedule(cfg, self.steps / cfg.total_steps)
            dist = disturbance_schedule(cfg, self.steps / cfg.total_steps)
            self.env.randomizer.scale = rand
            self.env.residual_scale = residual
            self.env.disturbance_scale = dist

            buf_obs = np.zeros((t_len, n, self.obs_dim), np.float32)
            buf_act = np.zeros((t_len, n, ACTION_DIM), np.float32)
            buf_logp = np.zeros((t_len, n), np.float32)
            buf_rew = np.zeros((t_len, n), np.float32)
            buf_done = np.zeros((t_len, n), np.float32)
            buf_val = np.zeros((t_len + 1, n), np.float32)
            t0 = time.time()

            for t in range(t_len):
                o = self.norm(obs)
                with torch.no_grad():
                    ot = torch.as_tensor(o, dtype=torch.float32)
                    d = self.model.dist(ot)
                    a = d.sample()
                    buf_logp[t] = d.log_prob(a).sum(-1).numpy()
                    buf_val[t] = self.model.value(ot).numpy()
                buf_obs[t] = o
                buf_act[t] = a.numpy()
                obs, rew, term, trunc, info = self.env.step(a.numpy())
                # 時間到被截斷 (不是真的結束)：用最後狀態的價值補回去
                if trunc.any():
                    with torch.no_grad():
                        fo = torch.as_tensor(self.norm(info["final_observation"][trunc]), dtype=torch.float32)
                        rew = rew.copy()
                        rew[trunc] += cfg.gamma * self.model.value(fo).numpy()
                buf_rew[t] = rew
                buf_done[t] = (term | trunc).astype(np.float32)
                self.norm.update(obs)
                for i in np.nonzero(info["done"])[0]:
                    self.recent["return"].append(info["episode_return"][i])
                    self.recent["success"].append(float(info["success"][i]))
                    self.recent["max_ct"].append(info["max_cross_track"][i])
                    if info["success"][i]:
                        self.recent["time"].append(info["episode_time"][i])
            with torch.no_grad():
                buf_val[t_len] = self.model.value(torch.as_tensor(self.norm(obs), dtype=torch.float32)).numpy()
            rollout_time = time.time() - t0

            # GAE
            adv = np.zeros((t_len, n), np.float32)
            last = np.zeros(n, np.float32)
            for t in reversed(range(t_len)):
                nonterminal = 1.0 - buf_done[t]
                delta = buf_rew[t] + cfg.gamma * buf_val[t + 1] * nonterminal - buf_val[t]
                last = delta + cfg.gamma * cfg.gae_lambda * nonterminal * last
                adv[t] = last
            ret = adv + buf_val[:t_len]

            stats = self.optimize(buf_obs.reshape(-1, self.obs_dim), buf_act.reshape(-1, ACTION_DIM),
                                  buf_logp.reshape(-1), adv.reshape(-1), ret.reshape(-1))
            self.steps += n * t_len
            self.update += 1

            elapsed = time.time() - start
            sps = self.steps / elapsed
            rec = {k: (float(np.mean(v)) if v else None) for k, v in self.recent.items()}
            self.recent = {k: [] for k in self.recent}
            self.log({"type": "train", "update": self.update, "steps": self.steps, "sps": sps, "stage": stage,
                      "randomize_scale": rand, "residual_scale": residual, "disturbance_scale": dist,
                      "episode_return": rec["return"], "success_rate": rec["success"],
                      "episode_time": rec["time"], "max_cross_track": rec["max_ct"],
                      "action_std": float(self.model.log_std.detach().exp().mean()),
                      "rollout_seconds": rollout_time, **stats})
            eta = (cfg.total_steps - self.steps) / max(sps, 1.0)
            self.status("training", stage=stage, sps=sps, eta_seconds=eta,
                        randomize_scale=rand, residual_scale=residual, disturbance_scale=dist)
            if self.update % cfg.eval_every == 0:
                self.evaluate()

        self.evaluate()
        self.save("final")
        self.status("done", elapsed_seconds=time.time() - start, best_score=self.best_score)
        self.metrics_file.close()

    def optimize(self, obs, act, logp_old, adv, ret):
        cfg = self.cfg
        obs = torch.as_tensor(obs)
        act = torch.as_tensor(act)
        logp_old = torch.as_tensor(logp_old)
        adv = torch.as_tensor(adv)
        ret = torch.as_tensor(ret)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        total = obs.shape[0]
        clip_fracs, pls, vls, kls = [], [], [], []
        for _ in range(cfg.epochs):
            perm = torch.randperm(total)
            for s in range(0, total, cfg.minibatch):
                idx = perm[s:s + cfg.minibatch]
                d = self.model.dist(obs[idx])
                logp = d.log_prob(act[idx]).sum(-1)
                ratio = (logp - logp_old[idx]).exp()
                pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * adv[idx]).mean()
                v_loss = ((self.model.value(obs[idx]) - ret[idx]) ** 2).mean()
                entropy = d.entropy().sum(-1).mean()
                loss = pg + cfg.value_coef * v_loss - cfg.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()
                with torch.no_grad():
                    kls.append(float((logp_old[idx] - logp).mean()))
                    clip_fracs.append(float(((ratio - 1).abs() > cfg.clip).float().mean()))
                pls.append(float(pg.detach()))
                vls.append(float(v_loss.detach()))
        return {"policy_loss": float(np.mean(pls)), "value_loss": float(np.mean(vls)),
                "approx_kl": float(np.mean(kls)), "clip_fraction": float(np.mean(clip_fracs))}

    # ---- 評估 ---------------------------------------------------------------
    def _eval_env(self, residual_scale=1.0):
        return SwerveAutoEnv(self.cfg.path, n_envs=self.cfg.eval_episodes, seed=10_000,
                             randomize_scale=1.0, residual_scale=residual_scale,
                             disturbance_scale=self.cfg.eval_disturbance, obs_version=self.cfg.obs_version)

    def evaluate_baseline(self) -> dict:
        """B1 = 目前的 PathPlanner (動作全 0)，在跟策略評估一樣的種子與完整隨機化下跑。"""
        env = self._eval_env()
        result = evaluate_policy(env, lambda o: np.zeros((o.shape[0], ACTION_DIM)), self.cfg.eval_episodes)
        nominal = evaluate_policy(SwerveAutoEnv(self.cfg.path, 4, seed=1, randomize_scale=0.0,
                                                obs_version=self.cfg.obs_version),
                                  lambda o: np.zeros((o.shape[0], ACTION_DIM)), 4)
        result["nominal_time"] = nominal["mean_time_success"]
        result["reference_time"] = float(env.traj.total_time)
        atomic_write(self.dir / "baseline.json", json.dumps(result, indent=2), required=True)
        self.record_trace("baseline_trace.json", lambda o: np.zeros((o.shape[0], ACTION_DIM)))
        return result

    def evaluate(self):
        _, residual, _ = schedule(self.cfg, self.steps / self.cfg.total_steps)
        env = self._eval_env(residual_scale=residual)
        r = evaluate_policy(env, lambda o: self.act(o, deterministic=True), self.cfg.eval_episodes)
        score = r["mean_return"]
        r.update({"type": "eval", "update": self.update, "steps": self.steps, "residual_scale": residual,
                  "score": score})
        self.log(r)
        self.record_trace("trace.json", lambda o: self.act(o, deterministic=True), residual)
        self.save("latest")
        if score > self.best_score:
            self.best_score = score
            self.save("best")
            atomic_write(self.dir / "best_eval.json", json.dumps(r, indent=2), required=True)

    def record_trace(self, filename, policy, residual_scale=1.0):
        """用標準參數 (不隨機) 跑一次，記下真實/估計位置與時間縮放，給監控網頁畫場地圖。"""
        # 標準參數、但有干擾 (固定種子 → 每次評估同一組碰撞/壓球，軌跡圖可以比較)
        env = SwerveAutoEnv(self.cfg.path, n_envs=1, seed=3, randomize_scale=0.0, residual_scale=residual_scale,
                            disturbance_scale=self.cfg.eval_disturbance, obs_version=self.cfg.obs_version)
        obs = env.reset()
        trace = {"t": [], "x": [], "y": [], "est_x": [], "est_y": [], "time_scale": [], "speed": [],
                 "cross_track": []}
        for _ in range(int(env.max_time / env.params.cycle_time_s) + 1):
            a = np.clip(policy(obs), -1, 1)
            obs, _, _, _, info = env.step(a)
            if info["done"][0]:
                break
            p, e = env.plant.pose[0], env.follower.odometry.pose[0]
            trace["t"].append(round(float(env.follower.time[0]), 3))
            trace["x"].append(round(float(p[0]), 4))
            trace["y"].append(round(float(p[1]), 4))
            trace["est_x"].append(round(float(e[0]), 4))
            trace["est_y"].append(round(float(e[1]), 4))
            trace["time_scale"].append(round(float(1.0 + 0.5 * a[0, 0]), 3))
            trace["speed"].append(round(float(np.hypot(*env.plant.velocity[0, :2])), 3))
            trace["cross_track"].append(round(float(env.ep_max_ct[0]), 4))
        trace["finished"] = bool(info["success"][0])
        trace["total_time"] = float(info["episode_time"][0])
        trace["path_x"] = [round(float(v), 4) for v in env.traj._ct_x[::5]]
        trace["path_y"] = [round(float(v), 4) for v in env.traj._ct_y[::5]]
        atomic_write(self.dir / filename, json.dumps(trace))

    # ---- 存檔 / 匯出 -------------------------------------------------------
    def save(self, tag: str):
        tmp = self.dir / f"{tag}.pt.tmp"
        torch.save({"model": self.model.state_dict(), "norm_mean": self.norm.mean, "norm_var": self.norm.var,
                    "config": asdict(self.cfg), "steps": self.steps}, tmp)
        for attempt in range(20):
            try:
                tmp.replace(self.dir / f"{tag}.pt")
                break
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
        export_policy(self.model, self.norm, self.dir / f"policy_{tag}.json", self.cfg)


def export_policy(model: ActorCritic, norm: RunningNorm, out: Path, cfg: TrainConfig):
    """匯出給機器人端 (Java) 推論的 JSON：觀測正規化 + actor 權重。只用平均值 (確定性策略)。"""
    layers = []
    for m in model.actor:
        if isinstance(m, nn.Linear):
            layers.append({"weight": m.weight.detach().numpy().round(7).tolist(),
                           "bias": m.bias.detach().numpy().round(7).tolist()})
    data = {
        "schemaVersion": 1,
        "path": cfg.path,
        "activation": "tanh",
        "obs_dim": int(norm.mean.shape[0]),
        "obs_version": cfg.obs_version,
        "action_dim": ACTION_DIM,
        "obs_mean": norm.mean.round(7).tolist(),
        "obs_std": np.sqrt(norm.var + 1e-8).round(7).tolist(),
        "obs_clip": 10.0,
        "layers": layers,
        "action_meaning": ["time_scale = 1 + 0.5*a0", "dvx_robot = 0.5*a1 m/s", "dvy_robot = 0.5*a2 m/s",
                           "domega = 1.0*a3 rad/s"],
    }
    atomic_write(out, json.dumps(data), required=True)
