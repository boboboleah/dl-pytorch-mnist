# DL �� PyTorch MNIST Pipeline

��С�ɸ���ʵѵ�����ݡ�ģ�͡�ѵ�������������ӻ���һ�����ܡ�

## Quickstart (CPU)
`ash
python -m venv .venv && .\.venv\Scripts\activate   # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# MLP
python src/train.py --epochs 3 --lr 1e-3 --dropout 0.2 --arch mlp --out logs/run1
python src/eval.py  --ckpt logs/run1/best.pth --arch mlp --out logs/run1
python src/plot_curves.py --log logs/run1/metrics.csv --out logs/run1

# CNN���Աȣ�
python src/train.py --epochs 3 --arch cnn --out logs/run2
python src/eval.py  --ckpt logs/run2/best.pth --arch cnn --out logs/run2
python src/plot_curves.py --log logs/run2/metrics.csv --out logs/run2
Repo Map
bash
���ƴ���
.
������ src/
��   ������ models.py          # MLP/CNN
��   ������ data.py            # MNIST dataloader
��   ������ train.py           # ѵ�������򣨱��� best.pth �� metrics.csv��
��   ������ eval.py            # ���Լ�������accuracy��
��   ������ plot_curves.py     # ѵ��/��֤���߻���
������ requirements.txt
������ docs/
��   ������ EXP_NOTES.md       # ʵ���¼/�Աȱ�ģ��
������ .github/workflows/ci.yml
Reproducibility
�̶�������ӣ���¼������ָ�굽 metrics.csv

С�������� CPU �Ͽ��ٸ��֣�����ѵ�����л� GPU��CUDA��

License
MIT

Results
MLP��


CNN �Աȣ�


Reproduce��ֻ�� MLP ����С���
bash
���ƴ���
python src/train.py --epochs 3 --lr 1e-3 --dropout 0.2 --arch mlp --out logs/run1
python src/eval.py  --ckpt logs/run1/best.pth --arch mlp --out logs/run1
python src/plot_curves.py --log logs/run1/metrics.csv --out logs/run1