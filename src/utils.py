"""Small helpers shared by every script and notebook."""
import os
import random
import numpy as np
import torch

SEED = 42

CLASS_NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End of speed limit 80", "Speed limit 100",
    "Speed limit 120", "No passing", "No passing (>3.5t)", "Right-of-way at junction",
    "Priority road", "Yield", "Stop", "No vehicles", "No vehicles >3.5t", "No entry",
    "General caution", "Curve left", "Curve right", "Double curve", "Bumpy road",
    "Slippery road", "Road narrows right", "Road work", "Traffic signals",
    "Pedestrians", "Children crossing", "Bicycles crossing", "Ice/snow",
    "Wild animals", "End of all limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Straight or right", "Straight or left", "Keep right", "Keep left",
    "Roundabout", "End of no passing", "End of no passing (>3.5t)",
]


def set_seed(seed: int = SEED):
    """Fix every random number generator so results can be reproduced."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
