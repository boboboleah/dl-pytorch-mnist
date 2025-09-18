
# DL — PyTorch MNIST Pipeline

> 最小可复现实训：数据→模型→训练→评估→可视化，一键可跑。

## Quickstart

```bash
git clone https://github.com/boboboleah/dl-pytorch-mnist.git
cd dl-pytorch-mnist

python -m venv .venv && source .venv/bin/activate  # Windows 用 .venv\Scripts\activate
pip install -r requirements.txt

# 训练 + 评估 + 画图
python src/train.py --epochs 3 --lr 1e-3 --dropout 0.2 --out logs/run1
python src/eval.py --ckpt logs/run1/best.pth --out logs/run1
python src/plot_curves.py --log logs/run1/metrics.csv --out logs/run1
```

## Repo Map
```
.
├── src/
│   ├── models.py          # MLP/CNN
│   ├── data.py            # MNIST dataloader
│   ├── train.py           # 训练主程序（保存 best.pth 与 metrics.csv）
│   ├── eval.py            # 测试集评估（accuracy）
│   └── plot_curves.py     # 训练/验证曲线绘制
├── requirements.txt
├── docs/
│   └── EXP_NOTES.md       # 实验记录/对比表模板
└── .github/workflows/ci.yml
```

## Reproducibility
- 固定随机种子、记录超参与指标到 `metrics.csv`
- 小批量可在 CPU 上快速复现，完整训练可切换 GPU（CUDA）

## License
MIT
