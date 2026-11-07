"""場上干擾：碰撞、推擠、牆與 HUB、地毯摩擦不均、輪胎磨耗、壓到球、卡球。

每個 episode 依 scale (0–1，課程學習用) 隨機抽出事件；每個控制週期 update() 把它們換成
RealisticPlant 的輸入：
- plant.mu_scale (N, 4)：每輪摩擦倍率 = 輪胎磨耗 × 地毯低摩擦區塊 × 壓到球
- plant.ext_force / ext_torque：碰撞衝量 (短時間大力)、被推擠 (持續力)、卡球阻力 (跟速度反向)
- plant.contact：牆與 HUB 的接觸力 (每個子步算，彈簧 + 阻尼，作用在保險桿角落上，會產生力矩)

沒有實機數據，大小都是估計值並隨機化：
- 碰撞：另一台機器人撞上來，0.1 秒內讓車速改變 0.3–1.2 m/s、角速度 ±1 rad/s
- 推擠：50–250 N 持續 0.3–1.5 秒
- 地毯區塊：0–3 塊、半徑 0.4–1.2 m、摩擦 ×0.5–0.85 (膠帶、地毯接縫、灰塵)
- 輪胎磨耗：每輪 ×0.85–1.0
- 壓到球：輪子被球墊高，0.15–0.5 秒內那一輪摩擦 ×0.15
- 卡球：球卡在保險桿下，30–120 N 阻力持續 0.5–2 秒
"""

from __future__ import annotations

import numpy as np

from .robot_software import robot_to_field

FIELD_X = 16.54
FIELD_Y = 8.07
HUB_CENTERS = np.array([[4.626, 4.035], [11.915, 4.035]])  # Constants.kBlue/RedHubCenter
HUB_HALF = 0.597  # HUB 約 47 in 見方 (估計)
ROBOT_HALF_LENGTH = 0.81 / 2  # PathPlanner settings：含保險桿
ROBOT_HALF_WIDTH = 0.861 / 2
CONTACT_K = 30_000.0  # N/m
CONTACT_C = 2_000.0  # N·s/m

# 保險桿上 8 個接觸點 (機器人座標)：四角 + 四邊中點
_CONTACT_POINTS = np.array([
    [ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH], [ROBOT_HALF_LENGTH, -ROBOT_HALF_WIDTH],
    [-ROBOT_HALF_LENGTH, ROBOT_HALF_WIDTH], [-ROBOT_HALF_LENGTH, -ROBOT_HALF_WIDTH],
    [ROBOT_HALF_LENGTH, 0.0], [-ROBOT_HALF_LENGTH, 0.0], [0.0, ROBOT_HALF_WIDTH], [0.0, -ROBOT_HALF_WIDTH],
])

MAX_BUMPS = 2
MAX_BALLS = 3
MAX_ZONES = 3
BUMP_DURATION = 0.1


def contact_forces(plant):
    """牆與 HUB 的接觸力 (場地座標) 與力矩。回傳 (fx, fy, torque)，形狀 (N,)。"""
    x, y, h = plant.pose[:, 0:1], plant.pose[:, 1:2], plant.pose[:, 2:3]
    vx, vy, w = plant.velocity[:, 0:1], plant.velocity[:, 1:2], plant.velocity[:, 2:3]
    ox, oy = robot_to_field(_CONTACT_POINTS[:, 0], _CONTACT_POINTS[:, 1], h)  # (N, 8)
    px, py = x + ox, y + oy
    # 接觸點速度 = 車體速度 + ω × r
    pvx, pvy = vx - w * oy, vy + w * ox

    nx = np.zeros_like(px)
    ny = np.zeros_like(px)
    depth = np.zeros_like(px)

    def add(pen, dir_x, dir_y):
        nonlocal nx, ny, depth
        deeper = pen > depth
        depth = np.where(deeper, pen, depth)
        nx = np.where(deeper, dir_x, nx)
        ny = np.where(deeper, dir_y, ny)

    add(-px, 1.0, 0.0)  # 左牆 x < 0
    add(px - FIELD_X, -1.0, 0.0)
    add(-py, 0.0, 1.0)
    add(py - FIELD_Y, 0.0, -1.0)
    for cx, cy in HUB_CENTERS:
        dx, dy = px - cx, py - cy
        inside = (np.abs(dx) < HUB_HALF) & (np.abs(dy) < HUB_HALF)
        pen_x = HUB_HALF - np.abs(dx)
        pen_y = HUB_HALF - np.abs(dy)
        use_x = pen_x < pen_y  # 從最近的邊推出去
        pen = np.where(inside, np.where(use_x, pen_x, pen_y), 0.0)
        add(pen, np.where(use_x, np.sign(dx), 0.0), np.where(use_x, 0.0, np.sign(dy)))

    v_normal = pvx * nx + pvy * ny
    magnitude = np.where(depth > 0.0, np.maximum(CONTACT_K * depth - CONTACT_C * v_normal, 0.0), 0.0)
    fx, fy = magnitude * nx, magnitude * ny
    torque = ox * fy - oy * fx
    plant.in_contact = np.any(depth > 0.0, axis=1)
    return fx.sum(axis=1), fy.sum(axis=1), torque.sum(axis=1)


class Disturbances:
    def __init__(self, n_envs: int, rng: np.random.Generator, trajectory, positions: np.ndarray):
        self.n = n_envs
        self.rng = rng
        self.traj = trajectory
        self.positions = positions  # 模組位置 (4, 2)
        n = n_envs
        self.bump_time = np.full((n, MAX_BUMPS), np.inf)
        self.bump_force = np.zeros((n, MAX_BUMPS, 2))
        self.bump_torque = np.zeros((n, MAX_BUMPS))
        self.push_start = np.full(n, np.inf)
        self.push_end = np.full(n, np.inf)
        self.push_force = np.zeros((n, 2))
        self.drag_start = np.full(n, np.inf)
        self.drag_end = np.full(n, np.inf)
        self.drag_force = np.zeros(n)
        self.zone_center = np.zeros((n, MAX_ZONES, 2))
        self.zone_radius = np.zeros((n, MAX_ZONES))  # 0 = 沒有這塊
        self.zone_scale = np.ones((n, MAX_ZONES))
        self.wear = np.ones((n, 4))
        self.ball_time = np.full((n, MAX_BALLS), np.inf)
        self.ball_end = np.full((n, MAX_BALLS), np.inf)
        self.ball_module = np.zeros((n, MAX_BALLS), dtype=int)
        self.contact_steps = np.zeros(n, dtype=int)
        self.events = np.zeros(n, dtype=int)  # 這個 episode 抽到幾個事件 (分析用)

    def reset(self, mask, scale: float, mass: np.ndarray, moi: np.ndarray):
        rng, n = self.rng, self.n
        total = self.traj.total_time
        idx = np.nonzero(mask)[0]
        k = len(idx)
        if k == 0:
            return

        def times(count_shape, lo=0.3, hi=0.9):
            return rng.uniform(lo * total, hi * total, count_shape)

        # 碰撞 (Poisson，平均 0.8·scale 次)
        nb = np.minimum(rng.poisson(0.8 * scale, k), MAX_BUMPS)
        bt = times((k, MAX_BUMPS))
        bt[np.arange(MAX_BUMPS)[None, :] >= nb[:, None]] = np.inf
        dv = rng.uniform(0.3, 1.2, (k, MAX_BUMPS))
        ang = rng.uniform(-np.pi, np.pi, (k, MAX_BUMPS))
        self.bump_time[idx] = bt
        self.bump_force[idx] = np.stack([np.cos(ang), np.sin(ang)], -1) * (dv * mass[idx, None] / BUMP_DURATION)[..., None]
        self.bump_torque[idx] = rng.uniform(-1.0, 1.0, (k, MAX_BUMPS)) * moi[idx, None] / BUMP_DURATION

        # 推擠 (機率 0.4·scale)
        has = rng.random(k) < 0.4 * scale
        start = times(k, 0.2, 0.8)
        self.push_start[idx] = np.where(has, start, np.inf)
        self.push_end[idx] = np.where(has, start + rng.uniform(0.3, 1.5, k), np.inf)
        ang = rng.uniform(-np.pi, np.pi, k)
        self.push_force[idx] = np.stack([np.cos(ang), np.sin(ang)], -1) * rng.uniform(50, 250, k)[:, None]

        # 卡球阻力 (機率 0.3·scale)
        has = rng.random(k) < 0.3 * scale
        start = times(k, 0.1, 0.8)
        self.drag_start[idx] = np.where(has, start, np.inf)
        self.drag_end[idx] = np.where(has, start + rng.uniform(0.5, 2.0, k), np.inf)
        self.drag_force[idx] = rng.uniform(30, 120, k)

        # 地毯低摩擦區塊：放在路線附近
        nz = np.minimum(rng.poisson(1.5 * scale, k), MAX_ZONES)
        s = rng.uniform(0, self.traj.length, (k, MAX_ZONES))
        cx = np.interp(s, self.traj._ct_s, self.traj._ct_x) + rng.normal(0, 0.4, (k, MAX_ZONES))
        cy = np.interp(s, self.traj._ct_s, self.traj._ct_y) + rng.normal(0, 0.4, (k, MAX_ZONES))
        self.zone_center[idx] = np.stack([cx, cy], -1)
        radius = rng.uniform(0.4, 1.2, (k, MAX_ZONES))
        radius[np.arange(MAX_ZONES)[None, :] >= nz[:, None]] = 0.0
        self.zone_radius[idx] = radius
        self.zone_scale[idx] = 1.0 - rng.uniform(0.15, 0.5, (k, MAX_ZONES)) * scale

        # 輪胎磨耗
        self.wear[idx] = 1.0 - rng.uniform(0.0, 0.15, (k, 4)) * scale

        # 壓到球 (Poisson，平均 1.0·scale 次)
        nballs = np.minimum(rng.poisson(1.0 * scale, k), MAX_BALLS)
        st = times((k, MAX_BALLS), 0.05, 0.95)
        st[np.arange(MAX_BALLS)[None, :] >= nballs[:, None]] = np.inf
        self.ball_time[idx] = st
        self.ball_end[idx] = st + rng.uniform(0.15, 0.5, (k, MAX_BALLS))
        self.ball_module[idx] = rng.integers(0, 4, (k, MAX_BALLS))

        self.contact_steps[idx] = 0
        self.events[idx] = nb + (self.push_start[idx] < np.inf) + (self.drag_start[idx] < np.inf) + nz + nballs

    def update(self, time: np.ndarray, plant):
        """依目前時間設定 plant 的摩擦倍率與外力 (每個控制週期呼叫一次)。"""
        t = time[:, None]
        # 摩擦倍率：磨耗 × 區塊 × 壓到球
        h = plant.pose[:, 2:3]
        ox, oy = robot_to_field(self.positions[:, 0], self.positions[:, 1], h)
        mx = plant.pose[:, 0:1] + ox  # 模組場地位置 (N, 4)
        my = plant.pose[:, 1:2] + oy
        scale = self.wear.copy()
        for z in range(MAX_ZONES):
            d = np.hypot(mx - self.zone_center[:, z, 0:1], my - self.zone_center[:, z, 1:2])
            inside = d < self.zone_radius[:, z:z + 1]
            scale = np.where(inside, scale * self.zone_scale[:, z:z + 1], scale)
        on_ball = (t >= self.ball_time) & (t < self.ball_end)  # (N, MAX_BALLS)
        for b in range(MAX_BALLS):
            hit = on_ball[:, b:b + 1] & (np.arange(4)[None, :] == self.ball_module[:, b:b + 1])
            scale = np.where(hit, scale * 0.15, scale)
        plant.mu_scale = scale

        # 外力
        bump_on = (t >= self.bump_time) & (t < self.bump_time + BUMP_DURATION)
        force = np.sum(self.bump_force * bump_on[..., None], axis=1)
        torque = np.sum(self.bump_torque * bump_on, axis=1)
        push_on = (time >= self.push_start) & (time < self.push_end)
        force = force + self.push_force * push_on[:, None]
        drag_on = (time >= self.drag_start) & (time < self.drag_end)
        speed = np.hypot(plant.velocity[:, 0], plant.velocity[:, 1])
        unit = plant.velocity[:, :2] / np.maximum(speed, 0.05)[:, None]
        force = force - unit * (self.drag_force * drag_on * np.minimum(speed / 0.05, 1.0))[:, None]
        plant.ext_force = force
        plant.ext_torque = torque
        if getattr(plant, "in_contact", None) is not None:
            self.contact_steps += plant.in_contact
