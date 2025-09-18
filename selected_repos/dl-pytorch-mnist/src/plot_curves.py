
import argparse, pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', type=str, required=True)
    ap.add_argument('--out', type=str, default='logs/run1')
    args = ap.parse_args()

    df = pd.read_csv(args.log)
    plt.figure()
    plt.plot(df['epoch'], df['train_loss'], label='train_loss')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.title('Train Loss'); plt.legend()
    plt.savefig(f"{args.out}/loss.png", dpi=150)

    plt.figure()
    plt.plot(df['epoch'], df['test_acc'], label='test_acc')
    plt.xlabel('epoch'); plt.ylabel('acc'); plt.title('Test Acc'); plt.legend()
    plt.savefig(f"{args.out}/acc.png", dpi=150)

if __name__ == '__main__':
    main()
