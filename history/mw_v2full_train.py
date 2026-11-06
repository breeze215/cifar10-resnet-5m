"""
NYU Tandon DL F26 - Efficient ResNet Architecture Challenge (CIFAR-10, <=5M params)
v4: same ResNet-26 student as v2/v3 (widths 64/128/256, blocks 4/4/3, 4.70M params), trained
from scratch with knowledge distillation from our own v2 and v3 models (also trained from scratch
on the permitted data only, as the competition rules allow).

Motivation: v2 (97.4% val, 0.900 public LB) and v3/AugMix (97.2% val, +3.2 pts on corrupted val,
0.894 public LB) suggest the hidden test set is a natural distribution shift, where accuracy tracks
clean accuracy. So v4 goes back to v2's augmentation and aims to raise clean accuracy.

Changes vs v3:
  - knowledge distillation (Hinton et al., 2015): the student also matches the averaged,
    temperature-softened predictions of the teacher models on the same augmented batch.
    The v2+v3 teacher ensemble scores 97.68% on our val split vs 97.40% for v2 alone.
  - augmentation back to v2's TrivialAugmentWide (--aug augmix keeps the v3 option)
  - RandomGrayscale removed: its only motivation was a statistic of the test images
  - label smoothing defaults to 0 when distilling (the teachers' soft targets replace it)
Kept from v3: robustness benchmark, --eval_ckpt, EMA of BN stats, mps support.

Teachers must not have trained on our validation images, or val accuracy is inflated. v2 and
v3 both used --val_size 5000 --seed 0, the same split as here.

Usage (Kaggle: first line of the cell is `%%writefile train_v4.py`, then `!python train_v4.py ...`):
    python train_v4.py --out_dir /kaggle/working --epochs 300        # teachers auto-found: /kaggle/input/**/*_teacher.pt
    python train_v4.py --teachers ckpt/v2_teacher.pt ckpt/v3_teacher.pt --data_dir data
    python train_v4.py --teachers                                    # no teachers = plain v2 recipe without grayscale
    python train_v4.py --eval_ckpt <path to model_ema.pt>            # benchmark only, no training
Outputs (in --out_dir): submission.csv, test_probs.npy, model.pt, model_ema.pt, train_log.csv, robustness.csv
"""
import argparse, os, sys, time, math, csv, copy, glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF

# ---------------------------------------------------------------- args
p = argparse.ArgumentParser()
p.add_argument("--data_dir", default=None, help="folder containing train_labels.csv (auto-found under /kaggle/input if omitted)")
p.add_argument("--out_dir", default=".")
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--batch_size", type=int, default=128)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--wd", type=float, default=5e-4)
p.add_argument("--warmup_epochs", type=int, default=5)
p.add_argument("--label_smoothing", type=float, default=None, help="default: 0.1 without teachers, 0 with")
p.add_argument("--cutmix_p", type=float, default=0.5, help="prob. of applying CutMix to a batch")
p.add_argument("--ema_decay", type=float, default=0.9995)
p.add_argument("--teachers", nargs="*", default=None,
               help="teacher state_dicts (ResNet 64/128/256, 4/4/3). Omit to auto-find /kaggle/input/**/*_teacher.pt; "
                    "pass the flag with no paths to train without distillation")
p.add_argument("--kd_alpha", type=float, default=0.9, help="weight of the distillation loss (rest goes to the label loss)")
p.add_argument("--kd_T", type=float, default=2.0,
               help="softmax temperature for distillation. 4 is the common default; our teachers were trained with "
                    "label smoothing, which already softens their outputs, so we use 2")
p.add_argument("--aug", choices=["augmix", "trivial"], default="trivial", help="per-image augmentation policy (trivial = v2)")
p.add_argument("--augmix_severity", type=int, default=3)
p.add_argument("--val_size", type=int, default=5000, help="0 = train on all 50k")
p.add_argument("--widths", type=int, nargs=3, default=[64, 128, 256])
p.add_argument("--blocks", type=int, nargs=3, default=[4, 4, 3])
# macOS starts DataLoader workers by re-running this (unguarded) script, which fails, so default
# to loading in the main process there. Linux (Kaggle) forks workers and is unaffected.
p.add_argument("--workers", type=int, default=0 if sys.platform == "darwin" else min(4, os.cpu_count() or 1))
p.add_argument("--seed", type=int, default=0)
p.add_argument("--max_train", type=int, default=0, help="debug: use only N training images")
p.add_argument("--eval_ckpt", default=None, help="skip training; run the robustness benchmark on this state_dict")
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
os.makedirs(args.out_dir, exist_ok=True)

if args.teachers is None:
    args.teachers = sorted(glob.glob("/kaggle/input/**/*_teacher.pt", recursive=True))
    assert args.teachers or args.eval_ckpt, \
        "no teachers found under /kaggle/input (*_teacher.pt); pass --teachers <paths>, or --teachers alone to disable"
if args.label_smoothing is None:
    args.label_smoothing = 0.0 if args.teachers else 0.1

# ---------------------------------------------------------------- data
if args.data_dir is None:
    hits = glob.glob("/kaggle/input/**/train_labels.csv", recursive=True)
    assert hits, "train_labels.csv not found under /kaggle/input; pass --data_dir"
    args.data_dir = os.path.dirname(hits[0])
print("data_dir:", args.data_dir, "| device:", dev, "| aug:", args.aug,
      "| label_smoothing:", args.label_smoothing, "| teachers:", args.teachers or "none")

def load_images(folder, filenames, cache):
    if os.path.exists(cache):
        return np.load(cache)
    arr = np.stack([np.asarray(Image.open(os.path.join(folder, f)).convert("RGB")) for f in filenames])
    np.save(cache, arr)
    return arr  # (N, 32, 32, 3) uint8

D = args.data_dir
train_df = pd.read_csv(os.path.join(D, "train_labels.csv"))
test_df = pd.read_csv(os.path.join(D, "test.csv"))
sample = pd.read_csv(os.path.join(D, "sample_submission.csv"))

t0 = time.time()
X = load_images(os.path.join(D, "train_images"), train_df["filename"], os.path.join(args.out_dir, "train_cache.npy"))
Xte = load_images(os.path.join(D, "test_images"), test_df["filename"], os.path.join(args.out_dir, "test_cache.npy"))
y = train_df["label"].values.astype(np.int64)
print(f"loaded train {X.shape} test {Xte.shape} in {time.time()-t0:.0f}s")

# Same split as v2 for the same seed (keep this the first use of np.random after seeding),
# so --eval_ckpt on a v2 checkpoint scores it on images it never trained on.
perm = np.random.permutation(len(X))
val_idx, tr_idx = perm[:args.val_size], perm[args.val_size:]
if args.max_train:
    tr_idx = tr_idx[:args.max_train]

# per-channel normalization stats (float64 to avoid accumulation error)
MEAN = torch.tensor(X[tr_idx].reshape(-1, 3).mean(0, dtype=np.float64) / 255, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
STD = torch.tensor(X[tr_idx].reshape(-1, 3).std(0, dtype=np.float64) / 255, dtype=torch.float32, device=dev).view(1, 3, 1, 1)

def to_input(u8):  # uint8 NCHW (any device) -> normalized float on dev, channels_last
    x = u8.to(dev, non_blocking=True).float().div_(255)
    return ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)

class TrainSet(Dataset):
    def __init__(self, imgs, labels, policy):
        self.x = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()  # uint8 NCHW
        self.y = torch.from_numpy(labels)
        self.tf = v2.Compose([
            v2.RandomCrop(32, padding=4, padding_mode="reflect"),
            v2.RandomHorizontalFlip(),
            policy,
        ])
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.tf(self.x[i]), self.y[i]

# TrivialAugmentWide (v2): one random op (colour, contrast, rotation, shear, ...) per image.
# AugMix (v3): blends the original with 3 random op chains; ~2x slower on Kaggle's CPUs.
policy = v2.AugMix(severity=args.augmix_severity) if args.aug == "augmix" else v2.TrivialAugmentWide()
train_loader = DataLoader(TrainSet(X[tr_idx], y[tr_idx], policy), batch_size=args.batch_size, shuffle=True,
                          drop_last=True, num_workers=args.workers, pin_memory=(dev == "cuda"),
                          persistent_workers=args.workers > 0)
Xva_u8 = torch.from_numpy(X[val_idx]).permute(0, 3, 1, 2).contiguous()
yva = torch.from_numpy(y[val_idx]).to(dev)
Xte_u8 = torch.from_numpy(Xte).permute(0, 3, 1, 2).contiguous()

def cutmix(x, y_soft):
    """Paste a random box from a shuffled copy of the batch; mix labels by area."""
    B, _, H, W = x.shape
    lam = np.random.beta(1.0, 1.0)
    rh, rw = int(H * math.sqrt(1 - lam)), int(W * math.sqrt(1 - lam))
    cy, cx = np.random.randint(H), np.random.randint(W)
    y1, y2 = max(cy - rh // 2, 0), min(cy + rh // 2, H)
    x1, x2 = max(cx - rw // 2, 0), min(cx + rw // 2, W)
    idx = torch.randperm(B, device=x.device)
    x = x.clone(); x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (H * W)
    return x, lam * y_soft + (1 - lam) * y_soft[idx]

# ---------------------------------------------------------------- robustness benchmark
# Corruption types from CIFAR-10-C (Hendrycks & Dietterich, 2019) at roughly its severity 3,
# applied to our own validation images. Brightness/contrast/saturation overlap with what the
# augmentations train on; noise/blur/pixelate/jpeg do not, so read them separately too.
CORRUPTIONS = ["gaussian_noise", "gaussian_blur", "pixelate", "jpeg", "brightness", "contrast", "saturate"]

def corrupt(u8, kind, seed=0):
    """uint8 NCHW (CPU) -> corrupted uint8 NCHW. Deterministic for a given seed."""
    if kind == "jpeg":
        return torch.stack([decode_jpeg(encode_jpeg(img, quality=20)) for img in u8])
    x = u8.float() / 255
    if kind == "gaussian_noise":
        x = x + 0.08 * torch.randn(x.shape, generator=torch.Generator().manual_seed(seed))
    elif kind == "gaussian_blur":
        x = TF.gaussian_blur(x, kernel_size=5, sigma=0.7)
    elif kind == "pixelate":    # downsample to 60% then blow back up with blocky pixels
        x = F.interpolate(F.interpolate(x, size=19, mode="area"), size=32, mode="nearest")
    elif kind == "brightness":
        x = x + 0.3
    elif kind == "contrast":    # squeeze each channel toward its per-image mean
        m = x.mean((2, 3), keepdim=True); x = (x - m) * 0.3 + m
    elif kind == "saturate":    # wash colours out toward grayscale
        x = TF.adjust_saturation(x, 0.3)
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)

def robustness(nets):
    """Val accuracy (no TTA) on clean + each corruption for each named net; prints and saves a table."""
    sets = {"clean": Xva_u8, **{k: corrupt(Xva_u8, k, args.seed) for k in CORRUPTIONS}}
    rows = []
    for name, net in nets:
        accs = {k: (predict(net, xs).argmax(1) == yva).float().mean().item() for k, xs in sets.items()}
        accs["corrupt_mean"] = float(np.mean([accs[k] for k in CORRUPTIONS]))
        rows += [(name, k, a) for k, a in accs.items()]
    print("robustness (val acc, no TTA):")
    for k in ["clean", *CORRUPTIONS, "corrupt_mean"]:
        print(f"  {k:15s} " + "  ".join(f"{name}={a:.4f}" for name, kk, a in rows if kk == k))
    with open(os.path.join(args.out_dir, "robustness.csv"), "w", newline="") as f:
        cw = csv.writer(f); cw.writerow(["net", "set", "val_acc"]); cw.writerows(rows)

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

ema = copy.deepcopy(model).eval()
for q in ema.parameters(): q.requires_grad_(False)

@torch.no_grad()
def ema_update(step):
    d = min(args.ema_decay, (1 + step) / (10 + step))
    for e, m in zip(ema.parameters(), model.parameters()):
        e.mul_(d).add_(m.detach(), alpha=1 - d)
    for e, m in zip(ema.buffers(), model.buffers()):
        if e.dtype.is_floating_point:   # BN running mean/var: average like the weights
            e.mul_(d).add_(m, alpha=1 - d)
        else:                           # BN num_batches_tracked counter
            e.copy_(m)

# ---------------------------------------------------------------- optim
decay = [q for n, q in model.named_parameters() if q.ndim > 1]
no_decay = [q for n, q in model.named_parameters() if q.ndim <= 1]   # BN + biases
opt = torch.optim.SGD([{"params": decay, "weight_decay": args.wd},
                       {"params": no_decay, "weight_decay": 0.0}],
                      lr=args.lr, momentum=0.9, nesterov=True)
steps_per_epoch = len(train_loader)
total, warm = args.epochs * steps_per_epoch, max(1, args.warmup_epochs * steps_per_epoch)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
use_amp = dev == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

def soft_ce(logits, target):
    return torch.sum(-target * F.log_softmax(logits.float(), dim=1), dim=1).mean()

@torch.no_grad()
def predict(net, X_u8, tta=False, bs=500):
    """Mean softmax over views. tta=True: 2 flips x 5 shifts (0, +-2px in x/y, reflect-padded)."""
    net.eval(); out = []
    shifts = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)] if tta else [(0, 0)]
    for k in range(0, len(X_u8), bs):
        x = to_input(X_u8[k:k + bs])
        xp = F.pad(x, (2, 2, 2, 2), mode="reflect")
        prob = 0
        for dy, dx in shifts:
            v = xp[:, :, 2 + dy:34 + dy, 2 + dx:34 + dx].contiguous(memory_format=torch.channels_last)
            for flip in (False, True):
                vv = v.flip(3) if flip else v
                with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
                    prob = prob + net(vv).float().softmax(1)
        out.append(prob / (2 * len(shifts)))
    return torch.cat(out)

# ---------------------------------------------------------------- benchmark-only mode
if args.eval_ckpt:
    assert len(val_idx), "--eval_ckpt needs a validation split (val_size > 0)"
    ema.load_state_dict(torch.load(args.eval_ckpt, map_location=dev))
    robustness([("ckpt", ema)])
    raise SystemExit

# ---------------------------------------------------------------- teachers (knowledge distillation)
TEACHER_WIDTHS, TEACHER_BLOCKS = [64, 128, 256], [4, 4, 3]   # v2/v3 architecture, independent of the student's
teachers = []
for path in args.teachers:
    t = ResNet(TEACHER_WIDTHS, TEACHER_BLOCKS).to(dev).to(memory_format=torch.channels_last)
    t.load_state_dict(torch.load(path, map_location=dev))
    t.eval()
    for q in t.parameters(): q.requires_grad_(False)
    teachers.append(t)
if teachers and len(val_idx):
    # Sanity check: these should match the teachers' own val accuracy (v2 0.9740, v3 0.9720 on
    # the seed-0 split). Much higher would mean a teacher trained on these val images.
    probs = [predict(t, Xva_u8) for t in teachers]
    accs = [f"{(pr.argmax(1) == yva).float().mean().item():.4f}" for pr in probs]
    ens = (sum(probs).argmax(1) == yva).float().mean().item()
    print(f"teacher val acc (no TTA): {accs} | averaged: {ens:.4f}")

@torch.no_grad()
def teacher_probs(xb):
    """Average of the teachers' temperature-softened probabilities for this (augmented) batch."""
    p = 0
    for t in teachers:
        with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
            p = p + F.softmax(t(xb).float() / args.kd_T, dim=1)
    return p / len(teachers)

def kd_loss(logits, target_probs):
    """KL(teacher || student) at temperature T, scaled by T^2 so its gradient size doesn't depend on T."""
    log_p = F.log_softmax(logits.float() / args.kd_T, dim=1)
    return F.kl_div(log_p, target_probs, reduction="batchmean") * args.kd_T ** 2

# ---------------------------------------------------------------- train
log = open(os.path.join(args.out_dir, "train_log.csv"), "w", newline="")
w = csv.writer(log); w.writerow(["epoch", "train_loss", "val_acc", "val_acc_ema", "lr", "sec"])
step = 0
for ep in range(args.epochs):
    model.train(); t0 = time.time(); tot_loss = 0.0; n = 0
    for xb_u8, yb in train_loader:
        xb = to_input(xb_u8); yb = yb.to(dev, non_blocking=True)
        ys = F.one_hot(yb, 10).float() * (1 - args.label_smoothing) + args.label_smoothing / 10
        if np.random.rand() < args.cutmix_p:
            xb, ys = cutmix(xb, ys)
        with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
            logits = model(xb)
        loss = soft_ce(logits, ys)
        if teachers:
            # Teachers see the exact same (CutMixed) images, so their soft predictions already
            # describe the mixed content; no label mixing needed for this term.
            loss = args.kd_alpha * kd_loss(logits, teacher_probs(xb)) + (1 - args.kd_alpha) * loss
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        ema_update(step); step += 1
        tot_loss += loss.item() * len(yb); n += len(yb)
    if len(val_idx):
        va = (predict(model, Xva_u8).argmax(1) == yva).float().mean().item()
        va_e = (predict(ema, Xva_u8).argmax(1) == yva).float().mean().item()
    else:
        va = va_e = float("nan")
    row = [ep + 1, tot_loss / n, va, va_e, sched.get_last_lr()[0], time.time() - t0]
    w.writerow(row); log.flush()
    print("ep {:3d} | loss {:.4f} | val {:.4f} | val_ema {:.4f} | lr {:.4f} | {:.1f}s".format(*row), flush=True)
log.close()
torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
torch.save(ema.state_dict(), os.path.join(args.out_dir, "model_ema.pt"))

# ---------------------------------------------------------------- final eval + submission (EMA weights)
if len(val_idx):
    for name, net in [("model", model), ("ema", ema)]:
        for tta in (False, True):
            acc = (predict(net, Xva_u8, tta=tta).argmax(1) == yva).float().mean().item()
            print(f"final val | {name:5s} | tta={str(tta):5s} | {acc:.4f}")
    robustness([("model", model), ("ema", ema)])
probs = predict(ema, Xte_u8, tta=True).cpu().numpy()
np.save(os.path.join(args.out_dir, "test_probs.npy"), probs)
id2pred = dict(zip(test_df["id"].values, probs.argmax(1)))
sub = pd.DataFrame({"id": sample["id"].values,
                    "label": [int(id2pred[i]) for i in sample["id"].values]})
assert len(sub) == len(sample) and sub["label"].between(0, 9).all()
sub.to_csv(os.path.join(args.out_dir, "submission.csv"), index=False)
print(f"wrote submission.csv ({len(sub)} rows), params {n_params:,}")
