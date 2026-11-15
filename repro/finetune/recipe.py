import torch
import torch.nn as nn
from timm.data import Mixup, create_transform
from torchvision import transforms

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
AA, REPROB, MIXUP, CUTMIX, SMOOTHING, DROP_PATH = "rand-m9-mstd0.5-inc1", 0.25, 0.8, 1.0, 0.1, 0.1


def train_transform():
    return create_transform(input_size=224, is_training=True, color_jitter=0.4, auto_augment=AA,
                            interpolation="bicubic", re_prob=REPROB, re_mode="pixel", re_count=1, mean=MEAN, std=STD)


def weak_transform():
    return transforms.Compose([
        transforms.RandomResizedCrop(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])


def make_mixup(num_classes=1000):
    return Mixup(mixup_alpha=MIXUP, cutmix_alpha=CUTMIX, prob=1.0, switch_prob=0.5, mode="batch",
                 label_smoothing=SMOOTHING, num_classes=num_classes)


class StochasticDepth(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p, self.active = float(p), False

    def forward(self, x):
        if not self.active or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep

    def extra_repr(self):
        return f"p={self.p}"


def set_drop_path(model, rate=DROP_PATH, names=("drop_path", "drop_path1", "drop_path2")):
    conts = [m for n, m in model.named_modules()
             if (n == "blocks" or n.endswith(".blocks")) and isinstance(m, (nn.ModuleList, nn.Sequential))]
    assert len(conts) == 1, [type(m).__name__ for m in conts]
    blocks, mods = list(conts[0]), []
    for i, blk in enumerate(blocks):
        found = False
        for name in names:
            old = getattr(blk, name, None)
            if isinstance(old, nn.Module):
                assert isinstance(old, nn.Identity) or type(old).__name__ == "DropPath", (name, type(old))
                sd = StochasticDepth(rate * i / max(1, len(blocks) - 1))
                setattr(blk, name, sd)
                mods.append(sd)
                found = True
        assert found, f"block {i} ({type(blk).__name__}) has no drop-path module"
    return mods


def drop_path_active(mods, active):
    for m in mods:
        m.active = active
