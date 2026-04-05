import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import torch
import torchvision
from torch import nn, Tensor
from torchvision.models import EfficientNet_B0_Weights, ResNet50_Weights, MobileNet_V3_Small_Weights

from acfg.modelconfig import ModelConfig
from lightning import LightningModule
from torch.nn import functional as F
from torch.optim import AdamW
from typing import Tuple


class MLPHead(nn.Module):
    def __init__(self, in_features: int, num_output_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(2048, num_output_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class PretrainedModelFactory:
    def __init__(self):
        self.available_models = {
            "efficientnet_b0": self._efficientnet_b0,
            "resnet_50": self._resnet_50,
            "vit_b_16": self._vit_b_16,
            "mobilenet_v3_small": self._mobilenet_v3_small,
        }

    @staticmethod
    def _freeze_pretrained_weights(model):
        for param in model.parameters():
            param.requires_grad = False

    @staticmethod
    def _efficientnet_b0():
        model = torchvision.models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        model.classifier = MLPHead(
            in_features=model.classifier[1].in_features,
            num_output_classes=ModelConfig.NUM_OUTPUT_CLASSES,
        )
        return model

    @staticmethod
    def _resnet_50():
        model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = MLPHead(
            in_features=model.fc.in_features,
            num_output_classes=ModelConfig.NUM_OUTPUT_CLASSES,
        )
        return model

    @staticmethod
    def _mobilenet_v3_small():
        model = torchvision.models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier = MLPHead(
            in_features=model.classifier[0].in_features,
            num_output_classes=ModelConfig.NUM_OUTPUT_CLASSES,
        )
        return model

    @staticmethod
    def _vit_b_16():
        raise NotImplementedError("ViT-B/16 not implemented yet.")

    def get_model(self, model_name: str) -> nn.Module:
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not available. Choose from {list(self.available_models.keys())}")
        return self.available_models[model_name]()


class DiseaseClassificationModel(nn.Module):
    def __init__(self, model_name: str) -> None:
        super().__init__()
        factory = PretrainedModelFactory()
        self.model = factory.get_model(model_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ClassificationModule(LightningModule):
    def __init__(self, model: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=1e-5)
        return optimizer

    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int) -> torch.Tensor:
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        self.log("TL", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        acc = self._accuracy(labels, outputs)
        self.log("VL", loss, prog_bar=True)
        self.log("VA", acc, prog_bar=True)
        return {"VL": loss, "VA": acc}

    @staticmethod
    def _accuracy(labels, outputs):
        preds = torch.argmax(outputs, dim=1)
        acc = torch.sum(preds == labels).float() / len(labels)
        return acc

    def test_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        self.log("TL", loss)