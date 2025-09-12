import numpy as np
from torchvision import transforms
from facenet_pytorch import fixed_image_standardization
import yaml
from PIL import Image
import torch
import os
from typing import Optional
from facenet_pytorch import InceptionResnetV1


# --- Configuration ---
config_file = 'config.yaml'
with open(config_file, 'r') as f:
    config = yaml.safe_load(f)

# --- Model & Device Setup ---
device = torch.device(config['MODEL']['device'])
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
print("FaceNet model loaded successfully on CPU.")
MODEL_IMAGE_SIZE = config['MODEL']['MODEL_IMAGE_SIZE'] # FaceNet model expects 160x160 images

def get_embedding(image_path, bbox):
    """Generates a FaceNet embedding for a given image file using a predefined bounding box."""
    if bbox is None:
        return None, f"Error processing {image_path}: No bounding box found."
    try:
        img = Image.open(image_path).convert('RGB')
        face_img = img.crop(bbox)
        face_tensor = preprocess(face_img).to(device) # type: ignore
        
        with torch.no_grad():
            embedding = resnet(face_tensor.unsqueeze(0))
            embedding = embedding.detach().cpu().numpy().flatten()
        return embedding, None
    except Exception as e:
        return None, f"Error processing {image_path}: {e}"


# --- Helper Function ---
def parse_bbox(bbox_str :str) -> Optional[tuple]:
    """Parses the bounding box string 'x1,y1,x2,y2' into a tuple of integers."""
    try:
        return tuple(map(int, bbox_str.split(',')))
    except (ValueError, AttributeError):
        return None

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cosine_distance(a, b):
    # distance = 1 - similarity
    return 1 - cosine_similarity(a, b)

# --- Preprocessing Pipeline ---
preprocess = transforms.Compose([
    transforms.Resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE)),
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])

# --- Simple Logging ---

os.makedirs(config['DIRECTORIES']['LOG_DIR'], exist_ok=True)
def simple_log(file :str, msg :Optional[str]):
    if msg:
        with open(file, 'a') as f:
            f.write(msg+'\n')
