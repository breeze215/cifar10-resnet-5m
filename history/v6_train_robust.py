"""
NYU Tandon DL F26 - Efficient ResNet Architecture Challenge (CIFAR-10, <=5M params)
v6: v5 + --aug robust (TrivialAugment|AugMix on CPU, photometric jitter/grayscale/blur on GPU) and a
"shifted val" scorecard: val accuracy under brightness/contrast/sharpness/blur/grayscale/JPEG/noise shifts,
also computed for reference checkpoints (--ref_models) so robustness can be compared without the leaderboard.
(v5:) WideResNet teachers. --arch wrn trains a large TEACHER (e.g. WRN-28-10, ~36M params) from
scratch on the allowed data; models over 5M params never produce test predictions or submission.csv.
Students (<=5M) distill from any mix of teachers via --teachers PATH... --teacher_specs SPEC...
Spec strings: "wrn-28-10", "resnet-4-4-3", "resnet-5-4-3-se".

Recipe: TrivialAugmentWide + RandomGrayscale + crop/flip, CutMix (p=0.5), label smoothing 0.1,
SGD-Nesterov + cosine, EMA weights, 10-view TTA. Trained from scratch.
--sam: Sharpness-Aware Minimization (Foret et al., 2021), rho=0.05; ~2x compute per step.
--teachers: model_ema.pt files (same architecture). Teachers see the exact same augmented/CutMix
batch as the student ("consistent" distillation, Beyer et al., 2022).
loss = kd_alpha * T^2 * KL(teacher_T || student_T) + (1 - kd_alpha) * CE(student, label)

Usage:
    robust:  python train_v6.py --blocks 5 4 3 --se --aug robust --epochs 600 [--val_size 0] [--ref_models a.pt b.pt --ref_specs resnet-5-4-3-se resnet-4-4-3]
    teacher: python train_v5.py --arch wrn --out_dir /kaggle/working/wrn --epochs 200 [--val_size 0]
    student: python train_v5.py --blocks 5 4 3 --se --teachers a.pt b.pt --teacher_specs wrn-28-10 resnet-4-4-3
Outputs (in --out_dir): submission.csv, test_probs.npy, model.pt, model_ema.pt, train_log.csv
"""
import argparse, os, time, math, csv, copy, glob
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2

# ---------------------------------------------------------------- args
p = argparse.ArgumentParser()
p.add_argument("--data_dir", default=None, help="folder containing train_labels.csv (auto-found under /kaggle/input if omitted)")
p.add_argument("--out_dir", default=".")
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--batch_size", type=int, default=128)
p.add_argument("--lr", type=float, default=0.1)
p.add_argument("--wd", type=float, default=5e-4)
p.add_argument("--warmup_epochs", type=int, default=5)
p.add_argument("--label_smoothing", type=float, default=0.1)
p.add_argument("--cutmix_p", type=float, default=0.5, help="prob. of applying CutMix to a batch")
p.add_argument("--ema_decay", type=float, default=0.9995)
p.add_argument("--val_size", type=int, default=5000, help="0 = train on all 50k")
p.add_argument("--widths", type=int, nargs=3, default=[64, 128, 256])
p.add_argument("--blocks", type=int, nargs=3, default=[4, 4, 3])
p.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
p.add_argument("--seed", type=int, default=0)
p.add_argument("--sam", action="store_true", help="use SAM (2 forward/backward passes per step)")
p.add_argument("--rho", type=float, default=0.05, help="SAM neighborhood radius")
p.add_argument("--cache_dir", default="/tmp", help="where decoded-image .npy caches go")
p.add_argument("--teachers", nargs="*", default=[], help="teacher model_ema.pt paths")
p.add_argument("--kd_alpha", type=float, default=0.9, help="weight of the distillation term")
p.add_argument("--kd_T", type=float, default=2.0, help="distillation temperature")
p.add_argument("--se", action="store_true", help="squeeze-and-excitation in every residual block")
p.add_argument("--se_reduction", type=int, default=16)
p.add_argument("--arch", choices=["resnet", "wrn"], default="resnet")
p.add_argument("--wrn_depth", type=int, default=28)
p.add_argument("--wrn_width", type=int, default=10)
p.add_argument("--teacher_specs", nargs="*", default=[], help="one spec per teacher, e.g. wrn-28-10 resnet-4-4-3 resnet-5-4-3-se")
p.add_argument("--aug", choices=["ta", "robust"], default="ta")
p.add_argument("--ref_models", nargs="*", default=[], help="checkpoints to score on clean + shifted val for comparison")
p.add_argument("--ref_specs", nargs="*", default=[])
p.add_argument("--max_train", type=int, default=0, help="debug: use only N training images")
args = p.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
os.makedirs(args.out_dir, exist_ok=True)

# ---------------------------------------------------------------- data
if args.data_dir is None:
    hits = glob.glob("/kaggle/input/**/train_labels.csv", recursive=True)
    assert hits, "train_labels.csv not found under /kaggle/input; pass --data_dir"
    args.data_dir = os.path.dirname(hits[0])
print("data_dir:", args.data_dir)

def load_images(folder, filenames, cache):
    if os.path.exists(cache):
        return np.load(cache)
    arr = np.stack([np.asarray(Image.open(os.path.join(folder, f)).convert("RGB")) for f in filenames])
    tmp = f"{cache}.{os.getpid()}.tmp.npy"            # unique per process: safe with parallel runs
    np.save(tmp, arr); os.replace(tmp, cache)
    return arr  # (N, 32, 32, 3) uint8

D = args.data_dir
train_df = pd.read_csv(os.path.join(D, "train_labels.csv"))
test_df = pd.read_csv(os.path.join(D, "test.csv"))
sample = pd.read_csv(os.path.join(D, "sample_submission.csv"))

t0 = time.time()
X = load_images(os.path.join(D, "train_images"), train_df["filename"], os.path.join(args.cache_dir, "train_cache.npy"))
Xte = load_images(os.path.join(D, "test_images"), test_df["filename"], os.path.join(args.cache_dir, "test_cache.npy"))
y = train_df["label"].values.astype(np.int64)
print(f"loaded train {X.shape} test {Xte.shape} in {time.time()-t0:.0f}s")

perm = np.random.permutation(len(X))
val_idx, tr_idx = perm[:args.val_size], perm[args.val_size:]
if args.max_train:
    tr_idx = tr_idx[:args.max_train]

# per-channel normalization stats (float64 to avoid accumulation error)
MEAN = torch.tensor(X[tr_idx].reshape(-1, 3).mean(0, dtype=np.float64) / 255, dtype=torch.float32, device=dev).view(1, 3, 1, 1)
STD = torch.tensor(X[tr_idx].reshape(-1, 3).std(0, dtype=np.float64) / 255, dtype=torch.float32, device=dev).view(1, 3, 1, 1)

def to_float(u8):  # uint8 NCHW -> float [0,1] on dev
    return u8.to(dev, non_blocking=True).float().div_(255)

def norm(x):
    return ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)

def to_input(u8):
    return norm(to_float(u8))

GRAY_W = torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)

def gpu_photometric(x):
    """Per-sample photometric augmentation on a float [0,1] batch (used with --aug robust):
    p=0.5 color jitter (brightness/contrast/saturation in [0.7, 1.3]), p=0.1 grayscale, p=0.1 Gaussian blur."""
    B = x.shape[0]
    u = lambda: torch.empty(B, 1, 1, 1, device=x.device).uniform_(0.7, 1.3)
    on = (torch.rand(B, 1, 1, 1, device=x.device) < 0.5).float()
    b, c, s_ = [1 + on * (f - 1) for f in (u(), u(), u())]
    x = x * b
    m = (x * GRAY_W).sum(1, keepdim=True).mean((2, 3), keepdim=True)
    x = (x - m) * c + m
    g = (x * GRAY_W).sum(1, keepdim=True)
    x = ((x - g) * s_ + g).clamp_(0, 1)
    gray = torch.rand(B, device=x.device) < 0.1
    if gray.any():
        x[gray] = (x[gray] * GRAY_W).sum(1, keepdim=True).expand(-1, 3, -1, -1)
    blur = torch.rand(B, device=x.device) < 0.1
    if blur.any():
        sig = float(np.random.uniform(0.3, 1.0))
        k = torch.exp(-torch.tensor([-1.0, 0.0, 1.0], device=x.device) ** 2 / (2 * sig ** 2)); k = k / k.sum()
        k2 = (k[:, None] * k[None, :]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        x[blur] = F.conv2d(F.pad(x[blur], (1, 1, 1, 1), mode="reflect"), k2, groups=3)
    return x

class TrainSet(Dataset):
    def __init__(self, imgs, labels):
        self.x = torch.from_numpy(imgs).permute(0, 3, 1, 2).contiguous()  # uint8 NCHW
        self.y = torch.from_numpy(labels)
        base = [v2.RandomCrop(32, padding=4, padding_mode="reflect"), v2.RandomHorizontalFlip()]
        if args.aug == "robust":   # + GPU photometric jitter/grayscale/blur in the training loop
            self.tf = v2.Compose(base + [v2.RandomChoice([v2.TrivialAugmentWide(), v2.AugMix(mixture_width=2)])])
        else:
            self.tf = v2.Compose(base + [
                v2.TrivialAugmentWide(),      # brightness/contrast/color/sharpness/geometry, 1 random op
                v2.RandomGrayscale(p=0.05),   # test set has ~3x more grayscale images than train
            ])
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.tf(self.x[i]), self.y[i]

train_loader = DataLoader(TrainSet(X[tr_idx], y[tr_idx]), batch_size=args.batch_size, shuffle=True,
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

# ---------------------------------------------------------------- model
class SE(nn.Module):
    """Squeeze-and-excitation (Hu et al., 2018): reweight channels by a gate computed from global context."""
    def __init__(self, c, r):
        super().__init__()
        h = max(8, c // r)
        self.fc1 = nn.Conv2d(c, h, 1); self.fc2 = nn.Conv2d(h, c, 1)
    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        return x * torch.sigmoid(self.fc2(F.relu(self.fc1(s))))

class BasicBlock(nn.Module):
    """y = ReLU(F(x) + S(x)),  F = conv3x3-BN-ReLU-conv3x3-BN[-SE],  S = identity or 1x1 conv-BN."""
    def __init__(self, cin, cout, stride, se=False, r=16):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False);     self.b2 = nn.BatchNorm2d(cout)
        self.se = SE(cout, r) if se else nn.Identity()
        self.short = nn.Identity()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.se(self.b2(self.c2(out)))
        return F.relu(out + self.short(x))

class ResNet(nn.Module):
    def __init__(self, widths, blocks, num_classes=10, se=False, r=16):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(inplace=True))
        layers, cin = [], widths[0]
        for s, (w, n) in enumerate(zip(widths, blocks)):
            for b in range(n):
                layers.append(BasicBlock(cin, w, 2 if (b == 0 and s > 0) else 1, se, r)); cin = w
        self.layers = nn.Sequential(*layers)       # 32x32 -> 16x16 -> 8x8
        self.fc = nn.Linear(cin, num_classes)
    def forward(self, x):
        x = self.layers(self.stem(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

class WideBlock(nn.Module):
    """Pre-activation wide residual block (Zagoruyko & Komodakis, 2016)."""
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin); self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout); self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.short = None if (stride == 1 and cin == cout) else nn.Conv2d(cin, cout, 1, stride, bias=False)
    def forward(self, x):
        o = F.relu(self.bn1(x))
        sc = x if self.short is None else self.short(o)
        o = self.c2(F.relu(self.bn2(self.c1(o))))
        return o + sc

class WRN(nn.Module):
    def __init__(self, depth, k, num_classes=10):
        super().__init__()
        n = (depth - 4) // 6; w = [16, 16 * k, 32 * k, 64 * k]
        self.stem = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        layers, cin = [], 16
        for c, s in zip(w[1:], [1, 2, 2]):
            for b in range(n):
                layers.append(WideBlock(cin, c, s if b == 0 else 1)); cin = c
        self.layers = nn.Sequential(*layers)
        self.bn = nn.BatchNorm2d(cin); self.fc = nn.Linear(cin, num_classes)
    def forward(self, x):
        x = F.relu(self.bn(self.layers(self.stem(x))))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

def make_model(spec):
    """'wrn-28-10' | 'resnet-4-4-3' | 'resnet-5-4-3-se'"""
    parts = spec.split("-")
    if parts[0] == "wrn":
        return WRN(int(parts[1]), int(parts[2]))
    se = parts[-1] == "se"
    blocks = [int(v) for v in parts[1:4]]
    return ResNet(args.widths, blocks, se=se, r=args.se_reduction)

student_spec = (f"wrn-{args.wrn_depth}-{args.wrn_width}" if args.arch == "wrn"
                else "resnet-" + "-".join(map(str, args.blocks)) + ("-se" if args.se else ""))
model = make_model(student_spec).to(dev).to(memory_format=torch.channels_last)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
over_budget = n_params > 5_000_000
print(f"model {student_spec} | trainable params: {n_params:,} | sam={args.sam} | val_size={args.val_size} | epochs={args.epochs}")
if over_budget:
    print("*** OVER THE 5M BUDGET: TEACHER ONLY. No test predictions or submission.csv will be written. ***")

# ---------------------------------------------------------------- teachers (frozen, not part of the final model)
teachers = []
specs = args.teacher_specs or [student_spec] * len(args.teachers)
assert len(specs) == len(args.teachers), "give one --teacher_specs entry per --teachers path"
for path, spec in zip(args.teachers, specs):
    t = make_model(spec)
    t.load_state_dict(torch.load(path, map_location="cpu"))
    t = t.to(dev).to(memory_format=torch.channels_last).eval()
    for q in t.parameters(): q.requires_grad_(False)
    teachers.append(t)
    print(f"teacher: {spec:16s} {sum(q.numel() for q in t.parameters()):>11,} params  {path}")

ema = copy.deepcopy(model).eval()
for q in ema.parameters(): q.requires_grad_(False)

@torch.no_grad()
def ema_update(step):
    d = min(args.ema_decay, (1 + step) / (10 + step))
    for e, m in zip(ema.parameters(), model.parameters()):
        e.mul_(d).add_(m.detach(), alpha=1 - d)
    for e, m in zip(ema.buffers(), model.buffers()):
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

def set_bn_momentum(net, on):
    """SAM: don't update BN running stats on the perturbed (2nd) forward pass."""
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            if on:
                m.momentum = getattr(m, "_m", m.momentum)
            else:
                m._m = m.momentum; m.momentum = 0.0

params = [q for q in model.parameters() if q.requires_grad]

def soft_ce(logits, target):
    return torch.sum(-target * F.log_softmax(logits.float(), dim=1), dim=1).mean()

@torch.no_grad()
def teacher_probs(x):
    """Ensemble teacher: mean of temperature-softened probabilities."""
    with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
        return torch.stack([(t(x).float() / args.kd_T).softmax(1) for t in teachers]).mean(0)

def total_loss(logits, ys, pt):
    ce = soft_ce(logits, ys)
    if pt is None:
        return ce
    logp_s = F.log_softmax(logits.float() / args.kd_T, dim=1)
    kd = (pt * (torch.log(pt.clamp_min(1e-8)) - logp_s)).sum(1).mean() * args.kd_T ** 2
    return args.kd_alpha * kd + (1 - args.kd_alpha) * ce

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

# ---------------------------------------------------------------- shifted-val scorecard
import torchvision.transforms.v2.functional as TF
from torchvision.io import encode_jpeg, decode_jpeg

def make_shifted(x):
    """Deterministic photometric/corruption shifts of the uint8 val set. JPEG and noise never appear in training."""
    g = torch.Generator().manual_seed(123)
    noisy = (x.float() + torch.randn(x.shape, generator=g) * 0.04 * 255).clamp(0, 255).to(torch.uint8)
    return {
        "bright+15%": TF.adjust_brightness(x, 1.15),
        "contrast+25%": TF.adjust_contrast(x, 1.25),
        "sharpen x2": TF.adjust_sharpness(x, 2.0),
        "blur s0.7": TF.gaussian_blur(x, [3, 3], [0.7, 0.7]),
        "grayscale": TF.rgb_to_grayscale(x, num_output_channels=3),
        "jpeg q60": torch.stack([decode_jpeg(encode_jpeg(im, quality=60)) for im in x]),
        "noise 0.04": noisy,
    }

SHIFTED = make_shifted(Xva_u8) if len(val_idx) else {}

def scorecard(net, label):
    accs = {"clean": (predict(net, Xva_u8).argmax(1) == yva).float().mean().item()}
    for k, xs in SHIFTED.items():
        accs[k] = (predict(net, xs).argmax(1) == yva).float().mean().item()
    shifted_mean = float(np.mean([v for k, v in accs.items() if k != "clean"]))
    print(f"SCORECARD {label:28s} clean {accs['clean']:.4f} | shifted-mean {shifted_mean:.4f} | " +
          " ".join(f"{k} {v:.4f}" for k, v in accs.items() if k != "clean"), flush=True)

if args.ref_models and len(val_idx):
    rspecs = args.ref_specs or [student_spec] * len(args.ref_models)
    assert len(rspecs) == len(args.ref_models)
    for path, spec in zip(args.ref_models, rspecs):
        r = make_model(spec); r.load_state_dict(torch.load(path, map_location="cpu"))
        r = r.to(dev).to(memory_format=torch.channels_last).eval()
        scorecard(r, f"ref {spec}")
        del r

if teachers and len(val_idx):   # what the teacher ensemble itself scores (upper reference)
    with torch.no_grad():
        tp = sum(predict(t, Xva_u8) for t in teachers)
    print(f"teacher ensemble val (flip TTA): {(tp.argmax(1) == yva).float().mean().item():.4f}")
    for t, path in zip(teachers, args.teachers):
        print(f"  single teacher val: {(predict(t, Xva_u8).argmax(1) == yva).float().mean().item():.4f}  {path}")

# ---------------------------------------------------------------- train
log = open(os.path.join(args.out_dir, "train_log.csv"), "w", newline="")
w = csv.writer(log); w.writerow(["epoch", "train_loss", "val_acc", "val_acc_ema", "lr", "sec"])
step = 0
for ep in range(args.epochs):
    model.train(); t0 = time.time(); tot_loss = 0.0; n = 0
    for xb_u8, yb in train_loader:
        xb = to_float(xb_u8)
        if args.aug == "robust":
            xb = gpu_photometric(xb)
        xb = norm(xb); yb = yb.to(dev, non_blocking=True)
        ys = F.one_hot(yb, 10).float() * (1 - args.label_smoothing) + args.label_smoothing / 10
        if np.random.rand() < args.cutmix_p:
            xb, ys = cutmix(xb, ys)
        pt = teacher_probs(xb) if teachers else None
        opt.zero_grad(set_to_none=True)
        with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
            logits = model(xb)
        loss = total_loss(logits, ys, pt)
        if args.sam:
            # 1) gradient at w (manually scaled for fp16 safety), 2) step to w + rho*g/|g|,
            # 3) gradient there, 4) restore w and update with that gradient.
            s = scaler.get_scale() if use_amp else 1.0
            (loss * s).backward()
            with torch.no_grad():
                gnorm = torch.norm(torch.stack([q.grad.float().norm() for q in params if q.grad is not None])) / s
                eps = []
                if torch.isfinite(gnorm):
                    k = args.rho / (gnorm + 1e-12) / s
                    for q in params:
                        e = None if q.grad is None else q.grad * k
                        if e is not None: q.add_(e)
                        eps.append(e)
            opt.zero_grad(set_to_none=True)
            set_bn_momentum(model, False)
            with torch.autocast(dev, dtype=torch.float16, enabled=use_amp):
                loss2 = total_loss(model(xb), ys, pt)
            scaler.scale(loss2).backward()
            set_bn_momentum(model, True)
            with torch.no_grad():
                for q, e in zip(params, eps):
                    if e is not None: q.sub_(e)
        else:
            scaler.scale(loss).backward()
        scaler.step(opt); scaler.update(); sched.step()
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
if len(val_idx):
    scorecard(ema, f"NEW {student_spec} ema")
if over_budget:
    print("teacher run finished: saved model.pt / model_ema.pt only (no test predictions).")
    raise SystemExit(0)
probs = predict(ema, Xte_u8, tta=True).cpu().numpy()
np.save(os.path.join(args.out_dir, "test_probs.npy"), probs)
id2pred = dict(zip(test_df["id"].values, probs.argmax(1)))
sub = pd.DataFrame({"id": sample["id"].values,
                    "label": [int(id2pred[i]) for i in sample["id"].values]})
assert len(sub) == len(sample) and sub["label"].between(0, 9).all()
sub.to_csv(os.path.join(args.out_dir, "submission.csv"), index=False)
print(f"wrote submission.csv ({len(sub)} rows), params {n_params:,}")
