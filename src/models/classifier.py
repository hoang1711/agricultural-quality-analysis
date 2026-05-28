import torch.nn as nn
import timm

CLASS_NAMES = [
    'freshapples', 'freshbanana', 'freshoranges',
    'rottenapples', 'rottenbanana', 'rottenoranges'
]

QUALITY_MAP = {
    'freshapples':  'fresh',
    'freshbanana':  'fresh',
    'freshoranges': 'fresh',
    'rottenapples':  'rotten',
    'rottenbanana':  'rotten',
    'rottenoranges': 'rotten'
}

class FruitQualityClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnetv2_s",
            pretrained=False,
            num_classes=0,
            global_pool="avg"
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))