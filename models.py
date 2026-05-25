import torch
import torch.nn as nn
import timm


class MosFPAD(nn.Module):
    def __init__(self, pretrained=True, num_classes=1):
        super().__init__()
        # Load MobileNetV1 pretrained
        self.model = timm.create_model("mobilenetv1_100", pretrained=pretrained)

        # Remove classification head
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()

        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
        )

        self.svc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = self.fc(x)
        x = self.svc(x)
        return x


def get_model(model_name: str, model_cfg: dict):
    if model_name == "mosfpad":
        return MosFPAD(
            pretrained=model_cfg.get("pretrained", True),
            num_classes=model_cfg.get("num_classes", 1),
        )

    raise ValueError(f"Unknown model name: {model_name}")
