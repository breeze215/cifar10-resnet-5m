"""Model definitions used by train.py, extracted verbatim so they can be imported
(e.g. by results.ipynb) without running training. results.ipynb asserts that this
code is identical to the corresponding block in train.py.

Final submitted model: make_model("resnet-5-4-3-se")  ->  4,810,754 trainable parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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

def make_model(spec, widths=(64, 128, 256), se_reduction=16):
    """'resnet-5-4-3-se' (final), 'resnet-4-4-3' (v2), 'wrn-28-10' (teacher, >5M, never submitted)."""
    parts = spec.split("-")
    if parts[0] == "wrn":
        return WRN(int(parts[1]), int(parts[2]))
    se = parts[-1] == "se"
    blocks = [int(v) for v in parts[1:4]]
    return ResNet(list(widths), blocks, se=se, r=se_reduction)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
