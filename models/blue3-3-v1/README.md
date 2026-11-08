# Blue3-3 RL 模型 v1

PPO 殘差策略，訓練 3000 萬步（最佳版本在 2420 萬步）。沒有學過碰撞、壓球、卡球（那是 v2）。

| 項目 | PathPlanner | RL v1 |
|---|---|---|
| 成功率（全程偏線 ≤ 30 cm） | 84.4% | **100%** |
| 平均完成時間 | 8.38 秒 | **7.06 秒（快 15.8%）** |
| 偏線 p95 | 30.6 cm | **14.9 cm** |
| 終點誤差 | 4.1 cm | **3.8 cm** |

評估：256 回合、完整領域隨機化（μ 0.6–1.2、延遲 0–3 週期等），無干擾。

| 檔案 | 用途 |
|---|---|
| `policy_best.json` | 機器人端（Java）推論用：觀測正規化 + actor 權重（2×64 tanh），觀測版本 1（24 維） |
| `best.pt` | PyTorch 檢查點：`python scripts/evaluate.py models/blue3-3-v1/best.pt` |
| `best_eval.json` / `baseline.json` | 上表的評估數字 |
| `config.json` | 訓練設定 |
