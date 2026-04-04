from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import models, transforms


DEFAULT_CLASS_NAMES = ["Type_1", "Type_2", "Type_3", "Type_4", "Type_5"]


@dataclass
class Prediction:
    label_index: int
    label_name: str
    confidence: float
    probabilities: np.ndarray


def resolve_device(device: str) -> torch.device:
    if device == "cuda":
        return torch.device("cuda")
    if device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(num_classes: int) -> torch.nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model


class TorchScrewClassifier:
    def __init__(self, checkpoint_path: Path, device: str = "auto") -> None:
        self.device = resolve_device(device)
        self.checkpoint_path = checkpoint_path

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}. "
                "Train the classifier first or pass --checkpoint_path."
            )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
        self.input_size = int(checkpoint.get("input_size", 224))
        self.model = build_model(len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
        std = checkpoint.get("std", [0.229, 0.224, 0.225])
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def _predict_tensor(self, image_bgr: np.ndarray) -> Prediction:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        label_index = int(np.argmax(probs))
        return Prediction(
            label_index=label_index,
            label_name=self.class_names[label_index],
            confidence=float(probs[label_index]),
            probabilities=probs,
        )

    def predict_with_tta(self, image_bgr: np.ndarray) -> Prediction:
        rotations = [image_bgr]
        rotations.append(cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE))
        rotations.append(cv2.rotate(image_bgr, cv2.ROTATE_180))
        rotations.append(cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE))

        probs = []
        for view in rotations:
            pred = self._predict_tensor(view)
            probs.append(pred.probabilities)
        mean_probs = np.mean(np.stack(probs, axis=0), axis=0)
        label_index = int(np.argmax(mean_probs))
        return Prediction(
            label_index=label_index,
            label_name=self.class_names[label_index],
            confidence=float(mean_probs[label_index]),
            probabilities=mean_probs,
        )
