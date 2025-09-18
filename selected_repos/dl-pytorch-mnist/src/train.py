
import argparse, os, random, csv
import torch, torch.nn as nn, torch.optim as optim
from data import get_loaders
from models import make_mlp, SmallCNN

def seed_all(s=42):
    import numpy as np, torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--dropout', type=float, default=0.2)
    ap.add_argument('--arch', type=str, default='mlp', choices=['mlp','cnn'])
    ap.add_argument('--out', type=str, default='logs/run1')
    args = ap.parse_args()

    seed_all(42)
    os.makedirs(args.out, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_loader, test_loader = get_loaders()
    model = make_mlp(args.dropout) if args.arch=='mlp' else SmallCNN(args.dropout)
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    best_acc = 0.0
    metrics_path = os.path.join(args.out, 'metrics.csv')
    with open(metrics_path, 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['epoch','train_loss','test_acc'])

    for ep in range(1, args.epochs+1):
        model.train()
        total = 0.0
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad(); out = model(x); loss = lossf(out,y); loss.backward(); opt.step()
            total += float(loss.item())
        # eval
        acc = evaluate(model, test_loader, device)
        with open(metrics_path, 'a', newline='') as f:
            csv.writer(f).writerow([ep, total/len(train_loader), acc])
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.out,'best.pth'))
        print(f"epoch {ep}: loss={total/len(train_loader):.4f} acc={acc:.4f}")

def evaluate(model, loader, device='cpu'):
    import torch
    model.eval(); ok=0; n=0
    with torch.no_grad():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=1)
            ok += (pred==y).sum().item(); n += y.numel()
    return ok/n

if __name__=='__main__':
    main()
