"""
Shared constants and preprocessing helpers.
Used by both train_model.py and streamlit_app.py so that training
and inference always agree on image size, normalization, and class order.
"""

import numpy as np
from PIL import Image

# EuroSAT (RGB) has 10 land cover classes. Order matters — it must match
# the order used by tensorflow_datasets' "eurosat/rgb" split, which is
# alphabetical by class folder name.
CLASS_NAMES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Short, human-friendly descriptions shown in the app under each prediction.
CLASS_DESCRIPTIONS = {
    "AnnualCrop": "Cropland with seasonal (annual) planting cycles.",
    "Forest": "Dense tree cover / woodland.",
    "HerbaceousVegetation": "Natural grassland or scrub, non-forested.",
    "Highway": "Roads and highway infrastructure.",
    "Industrial": "Factories, warehouses, industrial zones.",
    "Pasture": "Grazing land for livestock.",
    "PermanentCrop": "Orchards, vineyards, and other long-cycle crops.",
    "Residential": "Urban/suburban housing areas.",
    "River": "Rivers and other flowing water bodies.",
    "SeaLake": "Seas, lakes, and other large standing water bodies.",
}

IMG_SIZE = 64  # EuroSAT RGB images are natively 64x64


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL image into a normalized numpy array ready for the model.
    Returns shape (1, IMG_SIZE, IMG_SIZE, 3), values scaled to [0, 1].
    """
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr
