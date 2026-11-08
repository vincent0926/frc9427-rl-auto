# frc9427-rl-auto

FRC 9427 研究專題：用**強化學習（殘差策略）**讓 Swerve 自動路線跑得更快、又不偏離路線。

- 機器人端程式在另一個專案（PathPlanner + 自寫 Swerve）；這裡放模擬器、訓練與分析。
- 原則：**以無須實機數據為主**。參數取自程式碼常數與馬達規格，用寬範圍領域隨機化取代實機校正；實機只做最後驗收。

## 文件

| 文件 | 內容 |
|---|---|
| [docs/spec/rl-auto-spec.md](docs/spec/rl-auto-spec.md) | 專題規格書：目標、需求、安全流程、分期 |
| [docs/spec/rl-sim-design.md](docs/spec/rl-sim-design.md) | 模擬器設計：物理模型、軟體移植、隨機化、驗證 |

## 進度

依模擬器設計文件 §11 的實作順序：

- [x] 1. 機器人軟體移植（`rl_auto/sim/robot_software.py`）：座標轉換、模組目標、desaturate、模組最佳化、弧線里程計、PathPlanner 控制器
- [x] 2. PathPlanner 軌跡匯出（機器人專案的 `RlExportTest`）+ `trajectory.py`；Java 對照測試向量
- [x] 3. 理想物理（無打滑、無延遲）健全性檢查：`chassis.IdealPlant` + 閉迴路執行器 `rollout.PathFollower`
  - 6 條路線全部跑完：最大橫向誤差 ≤ 4.7 cm、終點誤差 ≤ 1.4 cm、里程計誤差 < 0.5 mm
  - 剩下的誤差來自機器人軟體本身（模組 cos 縮放、20 ms 離散週期），是 PathPlanner 基準的下限
  - 過程中找到並修正機器人程式的里程計誤差（見下方「發現」）
- [x] 4. 真實物理 `chassis.RealisticPlant`：TalonFX 速度迴路、Kraken 馬達與限流、輪胎滑移（摩擦圓）、載重轉移、電池壓降；理論健全性測試 `tests/test_physics.py`
  - 預設參數（μ 0.9）跑全部路線：最大橫向誤差 ≤ 14 cm；μ 0.6–0.7 時 Blue3-3 偏 36–90 cm，幾乎全是打滑造成的里程計漂移（尚未加視覺修正）
  - 發現：機器人設定的驅動 kV（2.35 V·s/m）比理論 1.98 高 19%、速度迴路 kP 幾乎 0 → 輪子會比命令快約 10%+，靠 PathPlanner 位置回授拉回
- [x] 5. 感測模型 `sensors.py`：視覺融合（`Swerve.addVisionMeasurement` + `PoseHistory` 移植：延遲補償、首筆對齊、10% 增益、2 m 跳動門檻、轉速 > 60°/s 不用）、抽象 Limelight（頻率、延遲、雜訊、掉幀）、陀螺儀漂移、命令延遲
  - 視覺能把打滑造成的終點誤差拉回來（μ 0.6 跑 Blue3-3：90 → 15 cm），但路線中途的偏移仍大（80 cm）：視覺每張只修 10%，追不上中途打滑。這正是 RL 要學的——在會打滑的地方放慢
- [x] 6. 向量化訓練環境 `env.SwerveAutoEnv`（自動重置、每個 episode 重抽領域隨機化 `randomization.py`）：1024 個環境約 5.7 萬步/秒
  - 動作 4 維：時間縮放 s ∈ [0.5, 1.5] + 機器人座標 Δvx、Δvy（±0.5 m/s）、Δω（±1 rad/s）；`residual_scale` 做課程學習
  - 觀測 24 維，只用機器人拿得到的資訊；獎勵與終止用真實位姿
  - 安全性質（有測試）：殘差單獨最多把車推離路線約 10 cm（PathPlanner kP 5 會拉回）
- [x] 7. 理論健全性檢查（`tests/test_physics.py`、`test_rollout.py`）

## 模擬基準（B1 = 目前的 PathPlanner，動作全 0）

256 個 episode，完整領域隨機化（μ 0.6–1.2、延遲 0–3 週期等）；成功 = 跑完且全程偏線 < 30 cm：

| 路線 | 無隨機化：成功率 / 時間 | 完整隨機化：成功率 / 時間 / 偏線 p95 |
|---|---|---|
| Blue3-3 | 100% / 8.38 s | 81% / 8.38 s / 30.6 cm |
| Blue1-3 | 100% / 9.00 s | 38% / 9.00 s / 31.6 cm |
| Blue3-3 固定加速 1.2× | — | 36% / 6.98 s / 32.3 cm |

盲目加速能省 17% 時間但成功率掉一半；RL 要學的是「在安全的地方才加速、會打滑的地方放慢」。

## 已訓練模型

| 模型 | 路線 | 成功率 | 時間 | 連結 |
|---|---|---|---|---|
| v1 | Blue3-3 | 100%（PathPlanner 84.4%） | 7.06 s（PathPlanner 8.38 s，快 15.8%） | [models/blue3-3-v1](https://github.com/vincent0926/frc9427-rl-auto/tree/main/models/blue3-3-v1) |

## 訓練與監控

```bash
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # 第一次
.venv/Scripts/python scripts/train.py --path Blue3-3 --steps 30000000 --name blue3-3-v1
.venv/Scripts/python scripts/dashboard.py        # 另一個終端機，打開 http://localhost:8765
.venv/Scripts/python scripts/evaluate.py runs/blue3-3-v1/best.pt --episodes 1024
```

- PPO（`rl_auto/train/ppo.py`）：策略 2×64 tanh（給 roboRIO 純 Java 推論）、觀測正規化一起匯出
- 課程學習：前 30% 只有時間縮放；隨機化範圍 30% → 100%（到一半時）；速度修正在 30%–70% 之間逐步開放
- 每 10 次更新用固定種子評估（256 回合、完整隨機化）並跟 PathPlanner 比；`best.pt` / `policy_best.json` 是評估分數最高的
- 監控網頁（`dashboard/`）每 3 秒更新：成功率、完成時間、偏線、獎勵、場地軌跡圖、時間縮放與車速、課程進度、訓練健康度
- `policy_*.json`：給機器人端的模型（觀測正規化 + 權重），`tests/test_ppo.py` 確認照 JSON 算出的動作跟 PyTorch 一致

## 下一步

- 安全監控器與 Java 推論（規格書 R4–R6）
