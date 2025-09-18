\# DL — PyTorch MNIST Pipeline



最小可复现实训：数据→模型→训练→评估→可视化，一键可跑。



\## Quickstart (CPU)

```bash

python -m venv .venv \&\& .\\.venv\\Scripts\\activate   # macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt



\# MLP

python src/train.py --epochs 3 --lr 1e-3 --dropout 0.2 --arch mlp --out logs/run1

python src/eval.py  --ckpt logs/run1/best.pth --arch mlp --out logs/run1

python src/plot\_curves.py --log logs/run1/metrics.csv --out logs/run1



\# CNN（对比）

python src/train.py --epochs 3 --arch cnn --out logs/run2

python src/eval.py  --ckpt logs/run2/best.pth --arch cnn --out logs/run2

python src/plot\_curves.py --log logs/run2/metrics.csv --out logs/run2

```



\## Repo Map

```

.

├── src/

│   ├── models.py          # MLP/CNN

│   ├── data.py            # MNIST dataloader

│   ├── train.py           # 训练主程序（保存 best.pth 与 metrics.csv）

│   ├── eval.py            # 测试集评估（accuracy）

│   └── plot\_curves.py     # 训练/验证曲线绘制

├── requirements.txt

├── docs/

│   └── EXP\_NOTES.md       # 实验记录/对比表模板

└── .github/workflows/ci.yml

```



\## Reproducibility

\- 固定随机种子，记录超参与指标到 `metrics.csv`

\- 小批量可在 CPU 上快速复现，完整训练可切换 GPU（CUDA）



\## License

MIT



\## Results



\*\*MLP：\*\*

!\[acc\_mlp](docs/acc.png)

!\[loss\_mlp](docs/loss.png)



\*\*CNN 对比：\*\*

!\[acc\_cnn](docs/acc\_cnn.png)

!\[loss\_cnn](docs/loss\_cnn.png)



\## Reproduce（只跑 MLP 的最小命令）

```bash

python src/train.py --epochs 3 --lr 1e-3 --dropout 0.2 --arch mlp --out logs/run1

python src/eval.py  --ckpt logs/run1/best.pth --arch mlp --out logs/run1

python src/plot\_curves.py --log logs/run1/metrics.csv --out logs/run1

```



