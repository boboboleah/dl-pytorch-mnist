
# 实验记录模板（示例）

| 试验 | 架构 | epoch | lr  | dropout | acc  | 备注 |
|------|------|-------|-----|---------|------|------|
| run1 | mlp  | 3     |1e-3 | 0.2     | 0.98 | baseline |
| run2 | cnn  | 3     |1e-3 | 0.2     | 0.99 | better |

- 将训练日志的 `metrics.csv` 配合 `plot_curves.py` 生成曲线图。
