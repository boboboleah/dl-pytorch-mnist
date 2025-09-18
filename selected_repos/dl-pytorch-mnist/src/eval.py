
import argparse, torch
from data import get_loaders
from models import make_mlp, SmallCNN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', type=str, required=True)
    ap.add_argument('--arch', type=str, default='mlp', choices=['mlp','cnn'])
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--device', type=str, default='cuda')
    ap.add_argument('--out', type=str, default='logs/run1')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu'
    model = make_mlp(args.dropout) if args.arch=='mlp' else SmallCNN(args.dropout)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.to(device)

    _, test_loader = get_loaders()
    model.eval(); ok=0; n=0
    with torch.no_grad():
        for x,y in test_loader:
            x,y = x.to(device), y.to(device)
            ok += (model(x).argmax(1)==y).sum().item(); n += y.numel()
    acc = ok/n
    print(f"Test acc: {acc:.4f}")
    with open(f"{args.out}/eval.txt","w") as f: f.write(f"acc={acc:.4f}\n")

if __name__=='__main__':
    main()
