from PIL import Image
import torch
from torchvision import transforms

from model import model, model_metadata, DEVICE


GREEK_CLASS_NAMES = {
    "ipeiros": "Ήπειρος",
    "kriti": "Κρήτη",
    "makedonia": "Μακεδονία",
    "nisia_aigaiou": "Νησιά Αιγαίου",
    "nisia_ioniou": "Νησιά Ιονίου",
    "peloponnisos": "Πελοπόννησος",
    "sterea_ellada": "Στερεά Ελλάδα",
    "thessalia": "Θεσσαλία",
    "thraki": "Θράκη",
}


image_transform = transforms.Compose([
    transforms.Resize((
        model_metadata["image_size"],
        model_metadata["image_size"]
    )),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=model_metadata["imagenet_mean"],
        std=model_metadata["imagenet_std"]
    )
])


def predict_image(image: Image.Image, top_k: int = 3):
    """
    Δέχεται μία εικόνα PIL και επιστρέφει τις Top-K προβλέψεις.
    """

    if image.mode != "RGB":
        image = image.convert("RGB")

    image_tensor = image_transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)

        top_probabilities, top_indices = torch.topk(
            probabilities,
            k=min(top_k, len(model_metadata["class_names"])),
            dim=1
        )

    results = []

    for probability, class_index in zip(
        top_probabilities[0],
        top_indices[0]
    ):
        class_name = model_metadata["class_names"][class_index.item()]

        results.append({
            "class_key": class_name,
            "class_name": GREEK_CLASS_NAMES.get(class_name, class_name),
            "confidence": round(probability.item() * 100, 2)
        })

    return results