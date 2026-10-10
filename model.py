from pathlib import Path

import torch
from torch import nn
from torchvision.models import convnext_tiny


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "trained"
    / "convnext_tiny_partial_fine_tuning"
    / "convnext_tiny_partial_fine_tuning.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model():
    """
    Φορτώνει το εκπαιδευμένο ConvNeXt Tiny και τα μεταδεδομένα του.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Δεν βρέθηκε το αρχείο του μοντέλου:\n{MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    class_names = checkpoint["class_names"]
    number_of_classes = checkpoint["number_of_classes"]

    model = convnext_tiny(weights=None)

    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features,
        number_of_classes
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(DEVICE)
    model.eval()

    metadata = {
        "class_names": class_names,
        "image_size": checkpoint["image_size"],
        "imagenet_mean": checkpoint["imagenet_mean"],
        "imagenet_std": checkpoint["imagenet_std"],
        "device": str(DEVICE)
    }

    return model, metadata


model, model_metadata = load_trained_model()