"""
Simple CNN backbone, reused as-is for:
  - condition A/B (in_channels=12, optical only)
  - condition C   (in_channels=14, optical+SAR concatenation fusion)
Per REQUIREMENTS.md #5/#6, the architecture is intentionally simple: this is the minimum-baseline
CNN, and the SAR-assisted "contribution" is the simplest listed fusion pattern (early channel
concatenation) -- not a novel architecture. Small enough to train on CPU in a few minutes/epoch
at this dataset size.
"""
import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
        self.features = nn.Sequential(
            block(in_channels, 32),   # 120 -> 60
            block(32, 64),            # 60 -> 30
            block(64, 128),           # 30 -> 15
            block(128, 256),          # 15 -> 7
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)  # logits, shape (B, num_classes)


def build_model(in_channels: int, num_classes: int) -> nn.Module:
    return SimpleCNN(in_channels, num_classes)


class TwoBranchCNN(nn.Module):
    """
    Second SAR-assisted fusion pattern (round 2), for an actual architecture comparison against
    the early-concatenation SimpleCNN(14ch) used in round 1: separate small optical and SAR
    encoders, each pooled to its own feature vector, concatenated, then a shared classifier head
    -- the "separate SAR and optical encoders followed by feature fusion" pattern explicitly
    listed as an acceptable option in REQUIREMENTS.md #6.

    Takes the SAME (14, H, W) concatenated input the dataset already produces for `mode="fusion"`
    (optical bands first, then SAR bands) and splits it internally, so no dataset/dataloader
    changes are needed to swap architectures.
    """

    def __init__(self, optical_channels: int = 12, sar_channels: int = 2, num_classes: int = 19,
                 feat_dim: int = 128):
        super().__init__()
        self.optical_channels = optical_channels
        self.sar_channels = sar_channels

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        # optical branch: same depth as the round-1 backbone (optical carries most of the signal)
        self.optical_enc = nn.Sequential(
            block(optical_channels, 32), block(32, 64), block(64, 128), block(128, feat_dim),
        )
        # SAR branch: shallower -- only 2 channels of genuinely lower-dimensional information
        self.sar_enc = nn.Sequential(
            block(sar_channels, 32), block(32, 64), block(64, feat_dim),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        optical = x[:, :self.optical_channels]
        sar = x[:, self.optical_channels:self.optical_channels + self.sar_channels]
        f_opt = self.pool(self.optical_enc(optical)).flatten(1)
        f_sar = self.pool(self.sar_enc(sar)).flatten(1)
        fused = torch.cat([f_opt, f_sar], dim=1)
        return self.classifier(fused)
