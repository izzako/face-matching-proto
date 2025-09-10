import numpy as np
from torchvision import transforms
from facenet_pytorch import fixed_image_standardization
import yaml

# --- Configuration ---
config_file = 'config.yaml'
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

MODEL_IMAGE_SIZE = config['MODEL']['MODEL_IMAGE_SIZE'] # FaceNet model expects 160x160 images

# --- Helper Function ---
def parse_bbox(bbox_str :str) -> tuple[int, ...]:
    """Parses the bounding box string 'x1,y1,x2,y2' into a tuple of integers."""
    try:
        return tuple(map(int, bbox_str.split(',')))
    except (ValueError, AttributeError):
        return None

def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm

# --- Preprocessing Pipeline ---
preprocess = transforms.Compose([
    transforms.Resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)),
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

# --- Simple Logging ---
def simple_log(file :str, msg :str):
    with open(file, 'a') as f:
        f.write(msg+'\n')
