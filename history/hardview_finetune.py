"""Continue the course project's v4 SE-ResNet from its own v4_val/model_ema.pt.

Use the same seed-0 45k/5k split as v4_val. The new treatment is discrete
hard-view mining, inspired by Wang et al., AugMax (NeurIPS 2021), section 4.4.
The network and checkpoint layout follow the project's existing v4 code.
Only the supplied training images and labels are used for fitting or selection.
"""
import argparse
import copy
import csv
import io
import math
import time
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F


class SE(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden = max(8, channels // 16)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, 1)
        return x * self.fc2(F.relu(self.fc1(pooled))).sigmoid()


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.se = SE(cout)
        self.short = nn.Identity()
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)
            )

    def forward(self, x):
        branch = self.se(self.b2(self.c2(F.relu(self.b1(self.c1(x))))))
        return F.relu(branch + self.short(x))


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        blocks = []
        cin = 64
        for stage, (width, depth) in enumerate(zip((64, 128, 256), (5, 4, 3))):
            for block in range(depth):
                blocks.append(BasicBlock(cin, width, 2 if stage and block == 0 else 1))
                cin = width
        self.layers = nn.Sequential(*blocks)
        self.fc = nn.Linear(cin, 10)

    def forward(self, x):
        return self.fc(self.layers(self.stem(x)).mean((2, 3)))


class DatasetFiles:
    def __init__(self, path):
        self.path = path
        self.archive = ZipFile(path) if path.is_file() else None

    def read(self, name):
        if self.archive is not None:
            return self.archive.read(name)
        return (self.path / name).read_bytes()

    def table(self, name):
        return list(csv.DictReader(io.StringIO(self.read(name).decode("utf-8-sig"))))

    def images(self, rows, folder):
        images = np.empty((len(rows), 32, 32, 3), dtype=np.uint8)
        for i, row in enumerate(rows):
            with Image.open(io.BytesIO(self.read(f"{folder}/{row['filename']}"))) as im:
                images[i] = np.asarray(im.convert("RGB"))
        return images


class Views:
    def __init__(self, device, mean, std):
        self.mean = torch.as_tensor(mean, device=device, dtype=torch.float32)[None, :, None, None]
        self.std = torch.as_tensor(std, device=device, dtype=torch.float32)[None, :, None, None]
        self.gray = torch.tensor([0.299, 0.587, 0.114], device=device)[None, :, None, None]
        self.coords = torch.arange(32, device=device)

    def normalize(self, x):
        return ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)

    def crop_flip(self, x):
        b = len(x)
        shifts = torch.randint(-4, 5, (2, b, 1), device=x.device)
        flips = torch.rand((b, 1), device=x.device) < 0.5
        rows = self.coords + shifts[0]
        cols = torch.where(flips, 31 - self.coords, self.coords) + shifts[1]
        rows = 31 - (31 - rows.abs()).abs()
        cols = 31 - (31 - cols.abs()).abs()
        batch = torch.arange(b, device=x.device)[:, None, None]
        return x[batch, rows[:, :, None], cols[:, None, :]].permute(0, 3, 1, 2).float() / 255

    def photo(self, x, bright, contrast, color, sharp):
        gray = (x * self.gray).sum(1, keepdim=True)
        centered = gray.mean((2, 3), keepdim=True)
        x = ((gray + color * (x - gray) - centered) * contrast + centered) * bright
        smooth = F.avg_pool2d(x, 3, stride=1, padding=1, count_include_pad=False)
        return (x + sharp * (x - smooth)).clamp_(0, 1)

    def random_photo(self, x):
        def rand(lo, hi):
            return torch.empty((len(x), 1, 1, 1), device=x.device).uniform_(lo, hi)

        color = rand(0.40, 1.35)
        color.masked_fill_(rand(0, 1) < 0.06, 0)
        return self.photo(x, rand(0.80, 1.22), rand(0.77, 1.30), color, rand(-0.24, 0.45))


def cutmix(x, y, rng):
    size = int(32 * math.sqrt(1 - rng.beta(1.0, 1.0)))
    cy, cx = rng.integers(0, 32, size=2)
    top, bottom = max(0, cy - size // 2), min(32, cy + size // 2)
    left, right = max(0, cx - size // 2), min(32, cx + size // 2)
    order = torch.randperm(len(x), device=x.device)
    x[:, :, top:bottom, left:right] = x[order, :, top:bottom, left:right]
    original = 1 - (bottom - top) * (right - left) / 1024
    return y * original + y[order] * (1 - original)


@torch.no_grad()
def hard_views(teacher, views, x, labels):
    a, b = views.random_photo(x), views.random_photo(x)
    images = views.normalize(torch.cat((a, b)))
    with torch.autocast(x.device.type, dtype=torch.float16, enabled=x.is_cuda):
        logits = teacher(images).float()
    loss_a, loss_b = F.cross_entropy(logits, labels.repeat(2), reduction="none").chunk(2)
    use_a = (loss_a > loss_b)[:, None, None, None]
    hardest = torch.where(use_a, a, b)
    # Keep one third of this batch on the ordinary crop/flip policy.
    changed = torch.rand((len(x), 1, 1, 1), device=x.device) < 2 / 3
    return torch.where(changed, hardest, x)


@torch.no_grad()
def update_ema(ema, model):
    for avg, current in zip(ema.parameters(), model.parameters()):
        avg.lerp_(current, 0.001)
    for avg, current in zip(ema.buffers(), model.buffers()):
        avg.copy_(current)


@torch.inference_mode()
def predict(net, images, views, batch_size, tta=False, shift=None):
    net.eval()
    offsets = [(0, 0), (2, 0), (-2, 0), (0, 2), (0, -2)] if tta else [(0, 0)]
    predictions = []
    for batch in images.split(batch_size):
        x = batch.permute(0, 3, 1, 2).float() / 255
        if shift is not None:
            x = views.photo(x, *shift)
        x = views.normalize(x)
        padded = F.pad(x, (2, 2, 2, 2), mode="reflect") if tta else x
        probs = 0
        for dy, dx in offsets:
            view = padded[:, :, 2 + dy:34 + dy, 2 + dx:34 + dx] if tta else x
            for image in (view, view.flip(3)):
                image = image.contiguous(memory_format=torch.channels_last)
                with torch.autocast(x.device.type, dtype=torch.float16, enabled=x.is_cuda):
                    probs = probs + net(image).float().softmax(1)
        predictions.append(probs / (2 * len(offsets)))
    return torch.cat(predictions)


def evaluate(model, images, labels, views, batch_size):
    conditions = (None, (1.13, 1.23, 0.8, 0.45), (0.90, 0.86, 0.40, -0.20))
    acc = [
        (predict(model, images, views, batch_size, shift=shift).argmax(1) == labels)
        .float().mean().item()
        for shift in conditions
    ]
    score = 0.5 * acc[0] + 0.25 * (acc[1] + acc[2])
    return score, acc


def export(model, files, views, batch_size, out):
    test_rows = files.table("test.csv")
    test = torch.from_numpy(files.images(test_rows, "test_images")).to(views.mean.device)
    probs = predict(model, test, views, batch_size, tta=True).cpu().numpy()
    predictions = dict(zip((row["id"] for row in test_rows), probs.argmax(1)))
    sample = files.table("sample_submission.csv")
    if len(predictions) != len(sample) or set(predictions) != {row["id"] for row in sample}:
        raise ValueError("Test and sample submission IDs do not match.")
    with (out / "submission.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(("id", "label"))
        writer.writerows((row["id"], int(predictions[row["id"]])) for row in sample)
    print(f"Wrote {out / 'submission.csv'} ({len(sample)} rows)", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, help="original v4_val/model_ema.pt")
    p.add_argument("--data", type=Path, help="competition data directory or ZIP")
    p.add_argument("--out", type=Path, default=Path("/kaggle/working/last_run"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--minutes", type=float, default=285, help="wall-time target including setup")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()
    start = time.monotonic()
    if args.epochs < 1 or args.batch_size < 2 or args.minutes <= 0:
        p.error("Use positive epochs and minutes and a batch size of at least two.")
    root = Path("/kaggle/input")
    if args.checkpoint is None:
        matches = sorted(root.rglob("v4_val/model_ema.pt"))
        if len(matches) != 1:
            p.error(f"Pass --checkpoint with the v4_val EMA file. Found {matches}")
        args.checkpoint = matches[0]
    if args.data is None:
        matches = sorted(root.rglob("train_labels.csv"))
        if len(matches) != 1:
            p.error(f"Pass --data with the competition data directory or ZIP. Found {matches}")
        args.data = matches[0].parent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        p.error("Enable a GPU accelerator before running the notebook.")
    torch.manual_seed(41)
    torch.backends.cudnn.benchmark = device.type == "cuda"
    rng = np.random.default_rng(41)
    args.out.mkdir(parents=True, exist_ok=True)

    net = ResNet().to(device=device, memory_format=torch.channels_last)
    count = sum(param.numel() for param in net.parameters() if param.requires_grad)
    assert count == 4_810_754 and count < 5_000_000
    net.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True))
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    print(f"{count:,} trainable parameters | {device} | {args.checkpoint}", flush=True)

    files = DatasetFiles(args.data)
    rows = files.table("train_labels.csv")
    if len(rows) != 50_000:
        raise ValueError("Expected 50,000 labeled images to reproduce the original v4 split.")
    perm = np.random.RandomState(0).permutation(len(rows))
    val_ids, train_ids = perm[:5000], perm[5000:]
    x_train = files.images([rows[i] for i in train_ids], "train_images")
    x_val = files.images([rows[i] for i in val_ids], "train_images")
    pixels = x_train.reshape(-1, 3)
    views = Views(device, pixels.mean(0, dtype=np.float64) / 255, pixels.std(0, dtype=np.float64) / 255)
    train, val = (torch.from_numpy(x).to(device) for x in (x_train, x_val))
    del pixels, x_train, x_val
    labels = torch.tensor([int(row["label"]) for row in rows], device=device)
    y_train, y_val = labels[train_ids.tolist()], labels[val_ids.tolist()]
    targets = F.one_hot(y_train, num_classes=10).float().mul_(0.95).add_(0.005)

    decay = [p for p in net.parameters() if p.ndim > 1]
    no_decay = [p for p in net.parameters() if p.ndim == 1]
    opt = torch.optim.SGD(
        [{"params": decay, "weight_decay": 0.0005}, {"params": no_decay}],
        lr=0.018, momentum=0.9, nesterov=True,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    baseline, original_acc = evaluate(ema, val, y_val, views, args.batch_size)
    print(f"v4 baseline | clean/bright/faded = {np.round(original_acc, 4)} | score {baseline:.4f}", flush=True)
    original_tta = (predict(ema, val, views, args.batch_size, tta=True).argmax(1) == y_val).float().mean().item()

    best_epoch, best_score, best_acc = 0, -math.inf, None
    best_path = args.out / "model_ema.pt"
    planned_epochs = args.epochs
    with (args.out / "train_log.csv").open("w", newline="") as log:
        writer = csv.writer(log)
        writer.writerow(("epoch", "loss", "clean", "bright", "faded", "score", "seconds"))
        writer.writerow((0, "", *original_acc, baseline, 0))
        for epoch in range(1, args.epochs + 1):
            ep_start = time.monotonic()
            progress = min(1, (epoch - 1) / max(1, planned_epochs - 1))
            lr = 0.018 * (0.10 + 0.90 * (1 + math.cos(math.pi * progress)) / 2)
            if epoch == 1:
                lr *= 0.35
            for group in opt.param_groups:
                group["lr"] = lr
            net.train()
            order = torch.randperm(len(train), device=device)
            order = order[:len(train) // args.batch_size * args.batch_size]
            loss_sum = torch.zeros((), device=device)
            for idx in order.split(args.batch_size):
                image, target = views.crop_flip(train[idx]), targets[idx]
                if rng.random() < 0.5:
                    image = hard_views(ema, views, image, y_train[idx])
                elif rng.random() < 0.5:
                    target = cutmix(image, target, rng)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                    logits = net(views.normalize(image))
                    loss = F.cross_entropy(logits.float(), target)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                update_ema(ema, net)
                loss_sum += loss.detach()
            mean_loss = loss_sum.item() / (len(order) // args.batch_size)
            ep_time = time.monotonic() - ep_start
            elapsed = time.monotonic() - start
            if epoch == 2:
                available = args.minutes * 60 - elapsed - 480
                planned_epochs = min(args.epochs, epoch + max(0, int(available / (ep_time * 1.10))))
                print(f"Measured {ep_time:.1f}s/epoch. Scheduled {planned_epochs} epochs.", flush=True)
            stop = epoch >= planned_epochs or elapsed + 2 * ep_time + 240 >= args.minutes * 60
            if epoch in (2, 8) or epoch % 10 == 0 or stop:
                score, acc = evaluate(ema, val, y_val, views, args.batch_size)
                # Favor checkpoints that retain v4's clean accuracy.
                rank = score - max(0.0, original_acc[0] - 0.0025 - acc[0])
                if rank > best_score:
                    best_epoch, best_score, best_acc = epoch, rank, acc
                    tmp = best_path.with_suffix(".tmp")
                    torch.save(ema.state_dict(), tmp)
                    tmp.replace(best_path)
                writer.writerow((epoch, mean_loss, *acc, score, ep_time))
                log.flush()
                print(f"epoch {epoch:3d} | loss {mean_loss:.4f} | "
                      f"clean/bright/faded {np.round(acc, 4)} | score {score:.4f} | {ep_time:.1f}s", flush=True)
            else:
                print(f"epoch {epoch:3d} | loss {mean_loss:.4f} | {ep_time:.1f}s", flush=True)
            if stop:
                break

    ema.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    final_tta = (predict(ema, val, views, args.batch_size, tta=True).argmax(1) == y_val).float().mean().item()
    print(f"Selected epoch {best_epoch} | clean/bright/faded {np.round(best_acc, 4)} | "
          f"clean 10-view TTA {original_tta:.4f} → {final_tta:.4f}", flush=True)
    export(ema, files, views, args.batch_size, args.out)


if __name__ == "__main__":
    main()
