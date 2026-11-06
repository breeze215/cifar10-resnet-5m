"""
NYU Tandon DL F26 - Efficient ResNet Architecture Challenge (CIFAR-10, <=5M params)
First submission: ResNet (3 stages, widths 80/160/320, 2 BasicBlocks per stage), trained from scratch.

Usage:
    python train.py --data_dir /path/to/competition_data --epochs 200
Outputs:
    submission.csv, model.pt, train_log.csv  (in --out_dir)
"""
import argparse, os, time, math, csv
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------- args
p = argparse.ArgumentParser()
p.add_argument("--data_dir", default="/kaggle/input")  # searched one level down
p.add_argument("--out_dir", default=".")
p.add_argument("--epochs", type=int, default=200)
p.add_argument("--batch_size", type=int, default=128)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--wd", type=float, default=5e-4)
p.add_argument("--warmup_epochs", type=int, default=5)
p.add_argument("--label_smoothing", type=float, default=0.1)
p.add_argument("--val_size", type=int, default=5000, help="0 = train on all 50k")
p.add_argument("--widths", type=int, nargs=3, default=[80, 160, 320])
p.add_argument("--blocks", type=int, nargs=3, default=[2, 2, 2])
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
os.makedirs(args.out_dir, exist_ok=True)

# ---------------------------------------------------------------- data
def find(name):
    """Locate a file/folder in data_dir or one level below it."""
    cands = [os.path.join(args.data_dir, name)] + \
            [os.path.join(args.data_dir, d, name) for d in sorted(os.listdir(args.data_dir))]
    for c in cands:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"{name} not found under {args.data_dir}")

def load_images(folder, filenames, cache):
    if os.path.exists(cache):
        return np.load(cache)
    arr = np.stack([np.asarray(Image.open(os.path.join(folder, f)).convert("RGB"))
                    for f in filenames])
    np.save(cache, arr)
    return arr  # (N, 32, 32, 3) uint8

train_df = pd.read_csv(find("train_labels.csv"))
test_df = pd.read_csv(find("test.csv"))
sample = pd.read_csv(find("sample_submission.csv"))

t0 = time.time()
X = load_images(find("train_images"), train_df["filename"], os.path.join(args.out_dir, "train_cache.npy"))
Xte = load_images(find("test_images"), test_df["filename"], os.path.join(args.out_dir, "test_cache.npy"))
y = train_df["label"].values.astype(np.int64)
print(f"loaded train {X.shape} test {Xte.shape} in {time.time()-t0:.0f}s")

# train/val split (stratified-ish via random perm)
perm = np.random.permutation(len(X))
val_idx, tr_idx = perm[:args.val_size], perm[args.val_size:]

def to_tensor(a):  # uint8 NHWC -> float NCHW on device, [0,1]
    return torch.from_numpy(a).to(dev).permute(0, 3, 1, 2).float().div_(255)

Xtr, ytr = to_tensor(X[tr_idx]), torch.from_numpy(y[tr_idx]).to(dev)
Xva, yva = to_tensor(X[val_idx]), torch.from_numpy(y[val_idx]).to(dev)
Xte_t = to_tensor(Xte)

mean = Xtr.mean(dim=(0, 2, 3), keepdim=True)
std = Xtr.std(dim=(0, 2, 3), keepdim=True)
norm = lambda x: (x - mean) / std

# ---------------------------------------------------------------- GPU augmentation
def _crop(xp, i, j, H, W):
    """Per-sample HxW crop of padded batch xp at offsets (i, j)."""
    B, C = xp.shape[:2]
    ar = torch.arange(H, device=xp.device)
    r = i[:, None] + ar                         # B,H
    c = j[:, None] + ar                         # B,W
    xp = xp.gather(2, r[:, None, :, None].expand(B, C, H, xp.shape[3]))
    return xp.gather(3, c[:, None, None, :].expand(B, C, H, W))

def augment(x):
    """Random crop (reflect pad 4) + horizontal flip + Cutout(16), batched on device. Input is normalized, so holes = mean pixel."""
    B, C, H, W = x.shape
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (B,), device=x.device)
    j = torch.randint(0, 9, (B,), device=x.device)
    x = _crop(xp, i, j, H, W)
    ar = torch.arange(H, device=x.device)
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    # Cutout: one 16x16 hole per image (center can be near edge)
    cy = torch.randint(0, H, (B,), device=x.device)
    cx = torch.randint(0, W, (B,), device=x.device)
    yy = ar[None, :, None]; xx = ar[None, None, :]
    mask = ((yy - cy[:, None, None]).abs() < 8) & ((xx - cx[:, None, None]).abs() < 8)
    return x * (~mask)[:, None].float()

# ---------------------------------------------------------------- model
class BasicBlock(nn.Module):
    """y = ReLU(F(x) + S(x)),  F = conv3x3-BN-ReLU-conv3x3-BN,  S = identity or 1x1 conv-BN."""
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False);     self.b2 = nn.BatchNorm2d(cout)
        self.short = nn.Identity()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return F.relu(out + self.short(x))

class ResNet(nn.Module):
    def __init__(self, widths, blocks, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        layers, cin = [], widths[0]
        for s, (w, n) in enumerate(zip(widths, blocks)):
            for b in range(n):
                layers.append(BasicBlock(cin, w, 2 if (b == 0 and s > 0) else 1)); cin = w
        self.layers = nn.Sequential(*layers)       # 32x32 -> 16x16 -> 8x8
        self.fc = nn.Linear(cin, num_classes)
    def forward(self, x):
        x = self.layers(self.stem(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

model = ResNet(args.widths, args.blocks).to(dev).to(memory_format=torch.channels_last)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"trainable params: {n_params:,}")
assert n_params <= 5_000_000, "over the 5M parameter budget"

# ---------------------------------------------------------------- optim
decay = [p for n, p in model.named_parameters() if p.ndim > 1]
no_decay = [p for n, p in model.named_parameters() if p.ndim <= 1]   # BN + biases
opt = torch.optim.SGD([{"params": decay, "weight_decay": args.wd},
                       {"params": no_decay, "weight_decay": 0.0}],
                      lr=args.lr, momentum=0.9, nesterov=True)
steps_per_epoch = math.ceil(len(Xtr) / args.batch_size)
total, warm = args.epochs * steps_per_epoch, args.warmup_epochs * steps_per_epoch
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
use_amp = dev == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

@torch.no_grad()
def predict(Xs, bs=1000):
    """Softmax probs averaged over original + horizontal flip (TTA)."""
    model.eval(); out = []
    for k in range(0, len(Xs), bs):
        xb = norm(Xs[k:k + bs]).contiguous(memory_format=torch.channels_last)
        with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
            out.append((model(xb).float().softmax(1) + model(xb.flip(3)).float().softmax(1)))
    return torch.cat(out)

# ---------------------------------------------------------------- train
log = open(os.path.join(args.out_dir, "train_log.csv"), "w", newline="")
w = csv.writer(log); w.writerow(["epoch", "train_loss", "train_acc", "val_acc", "lr", "sec"])
for ep in range(args.epochs):
    model.train(); t0 = time.time()
    idx = torch.randperm(len(Xtr), device=dev)
    tot_loss = tot_corr = 0
    for k in range(0, len(Xtr), args.batch_size):
        b = idx[k:k + args.batch_size]
        xb = augment(norm(Xtr[b])).contiguous(memory_format=torch.channels_last); yb = ytr[b]
        with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
            logits = model(xb); loss = crit(logits, yb)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot_loss += loss.item() * len(b); tot_corr += (logits.argmax(1) == yb).sum().item()
    val_acc = (predict(Xva).argmax(1) == yva).float().mean().item() if len(Xva) else float("nan")
    row = [ep + 1, tot_loss / len(Xtr), tot_corr / len(Xtr), val_acc, sched.get_last_lr()[0], time.time() - t0]
    w.writerow(row); log.flush()
    print("ep {:3d} | loss {:.4f} | train {:.4f} | val {:.4f} | lr {:.4f} | {:.1f}s".format(*row))
log.close()
torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))

# ---------------------------------------------------------------- submission
pred = predict(Xte_t).argmax(1).cpu().numpy()
id2pred = dict(zip(test_df["id"].values, pred))
sub = pd.DataFrame({"id": sample["id"].values,
                    "label": [int(id2pred[i]) for i in sample["id"].values]})
assert len(sub) == len(sample) and sub["label"].between(0, 9).all()
sub.to_csv(os.path.join(args.out_dir, "submission.csv"), index=False)
print(f"wrote submission.csv ({len(sub)} rows), params {n_params:,}")
