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
- [ ] 2. PathPlanner 軌跡匯出（Java 工具）+ `trajectory.py`
- [ ] 3. 理想物理（無打滑、無延遲）健全性檢查
- [ ] 4. 馬達模型、輪胎打滑、載重轉移
- [ ] 5. 感測模型（里程計漂移、視覺）
- [ ] 6. 向量化環境 + 效能測試
- [ ] 7. 理論健全性檢查

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
