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
- [ ] 6. 向量化環境 + 效能測試（目前 1024 環境約 1.5 萬步/秒，目標 5 萬；瓶頸是每步對整條路線算橫向誤差）
- [ ] 7. 理論健全性檢查

## 發現（已回饋到機器人程式）

| 日期 | 發現 | 處理 |
|---|---|---|
| 2026-11-05 | 模組速度上限 4.0 m/s 會截掉路線要求的 5.0 m/s，自動一路落後 | 改用理論上限 `MaxModuleSpeedMps ≈ 4.64` |
| 2026-11-05 | PathPlanner `driveCurrentLimit` ≤ 80 A 時生成 0 秒的壞軌跡 | 維持 95 A，加常駐測試擋住 |
| 2026-11-05 | 里程計用週期**結束**的朝向轉換位移，邊走邊轉時累積誤差（Blue3-3 約 6 cm，不打滑也會） | 改用週期中間朝向，誤差 < 0.5 mm |
| 2026-11-05 | Blue3-1、Blue3-3 的起點速度設 4.0 m/s，但自動第一段 / 射擊後是從靜止開始 | 實機 FollowPathCommand 會重新生成，不影響實機；匯出工具改成照自動流程判斷 |

## 機器人匯出資料

`data/robot-export/` 由機器人專案的 `RlExportTest` 產生，**不要手改**：

- `trajectories/*.csv`：每條 PathPlanner 路線用同一份 RobotConfig 生成的軌跡（5 ms 取樣）。自動的第一段或接在射擊後面的路線從靜止生成（同 FollowPathCommand），接在另一條路線後面的用理想軌跡；`_summary.csv` 有各路線總時間與是否從靜止開始
- `vectors/*.csv`：SwerveMath 與 `SwerveModule.setTarget` 的輸入輸出，`tests/test_java_vectors.py` 用來確認 Python 移植跟 Java 一致

機器人程式或路線改了之後重新匯出（在機器人專案資料夾，Git Bash）：

```bash
RL_EXPORT=1 ./gradlew test --offline --tests "*RlExportTest*" --rerun
cp -r build/rl-export/* <這個專案>/data/robot-export/
```

## 開發

需要 Python 3.10 以上。

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"   # Windows；macOS/Linux 用 .venv/bin/python
.venv/Scripts/python -m pytest
```

所有內部計算使用 SI 單位（m、s、kg、V、rad）。移植的機器人邏輯必須跟 Java 版行為一致；改機器人程式時要同步更新 `robot_software.py` 和測試。

## 安全

模擬器與策略**不直接連線機器人**。上機測試照規格書 §7 的安全流程，急停一律用 Driver Station。

## 授權

MIT，見 [LICENSE](LICENSE)。歡迎其他隊伍參考使用。
