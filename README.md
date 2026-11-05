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
- [ ] 3. 理想物理（無打滑、無延遲）健全性檢查
- [ ] 4. 馬達模型、輪胎打滑、載重轉移
- [ ] 5. 感測模型（里程計漂移、視覺）
- [ ] 6. 向量化環境 + 效能測試
- [ ] 7. 理論健全性檢查

## 機器人匯出資料

`data/robot-export/` 由機器人專案的 `RlExportTest` 產生，**不要手改**：

- `trajectories/*.csv`：每條 PathPlanner 路線用同一份 RobotConfig 生成的軌跡（5 ms 取樣）；`_summary.csv` 是各路線總時間
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
