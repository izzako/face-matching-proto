import torch
from facenet_pytorch import InceptionResnetV1
import numpy as np
import pandas as pd
import os
from PIL import Image
import random
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import matplotlib.pyplot as plt
import custom_utils as cu

random.seed(42)

# --- Configuration ---
CSV_FILE = cu.config['DIRECTORIES']['CSV_FILE']
IMAGE_DIR = cu.config['DIRECTORIES']['IMAGE_DIR']
NUM_PAIRS = cu.config['APP']['EVAL_NUM_PAIRS'] # Number of each genuine and imposter pairs to test.

# --- Model & Device Setup ---
device = torch.device(cu.config['MODEL']['device'])
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
print("FaceNet model loaded successfully on CPU.")


def get_embedding(image_path, bbox):
    """Generates a FaceNet embedding for a given image file using a predefined bounding box."""
    if bbox is None:
        return None
    try:
        img = Image.open(image_path).convert('RGB')
        face_img = img.crop(bbox)
        face_tensor = cu.preprocess(face_img).to(device)
        
        with torch.no_grad():
            embedding = resnet(face_tensor.unsqueeze(0))
        return cu.normalize_embedding(embedding.detach().cpu().numpy().flatten())
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def generate_evaluation_pairs(df, image_dir, num_pairs):
    """Generates balanced genuine and imposter pairs based on available images."""
    # 1. Create a lookup for bboxes from the DataFrame
    df['lookup_key'] = df['name'].str.replace(' ', '_') + '_' + df['image_id'].astype(str)
    bbox_lookup = df.set_index('lookup_key')['bbox'].to_dict()

    # 2. Scan available images and group them by person
    person_to_images = {}
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    for filename in image_files:
        try:
            # Filename is like 'Name_Lastname_imageid.jpg'
            parts = filename.rsplit('_', 1)
            name = parts[0].replace('_', ' ')
            if name not in person_to_images:
                person_to_images[name] = []
            person_to_images[name].append(filename)
        except IndexError:
            continue # Skip malformed filenames

    # 3. Filter for people with at least two available images for genuine pairs
    people_with_multiple_images = {name: files for name, files in person_to_images.items() if len(files) > 1}
    
    genuine_pairs = []
    imposter_pairs = []
    
    # --- Generate Genuine Pairs ---
    print(f"Generating {num_pairs} genuine pairs from available images...")
    if not people_with_multiple_images:
        print("Warning: Not enough people with multiple images to generate genuine pairs.")
        return [], []
        
    people_list_genuine = list(people_with_multiple_images.keys())
    while len(genuine_pairs) < num_pairs and people_list_genuine:
        person_name = random.choice(people_list_genuine)
        file1, file2 = random.sample(people_with_multiple_images[person_name], 2)
        
        path1 = os.path.join(image_dir, file1)
        path2 = os.path.join(image_dir, file2)
        
        bbox1_str = bbox_lookup.get(os.path.splitext(file1)[0])
        bbox2_str = bbox_lookup.get(os.path.splitext(file2)[0])
        
        if bbox1_str and bbox2_str:
            genuine_pairs.append(((path1, bbox1_str), (path2, bbox2_str)))
    
    # --- Generate Imposter Pairs ---
    print(f"Generating {num_pairs} imposter pairs from available images...")
    people_list_imposter = list(person_to_images.keys())
    if len(people_list_imposter) < 2:
        print("Warning: Not enough unique people to generate imposter pairs.")
        return genuine_pairs, []

    while len(imposter_pairs) < num_pairs and len(people_list_imposter) >= 2:
        name1, name2 = random.sample(people_list_imposter, 2)
        
        file1 = random.choice(person_to_images[name1])
        file2 = random.choice(person_to_images[name2])
        
        path1 = os.path.join(image_dir, file1)
        path2 = os.path.join(image_dir, file2)
        
        bbox1_str = bbox_lookup.get(os.path.splitext(file1)[0])
        bbox2_str = bbox_lookup.get(os.path.splitext(file2)[0])
        
        if bbox1_str and bbox2_str:
            imposter_pairs.append(((path1, bbox1_str), (path2, bbox2_str)))
            
    return genuine_pairs, imposter_pairs


# --- Main Evaluation Logic ---
def main():
    try:
        df = pd.read_csv(CSV_FILE)

        # Filter out images that are not available
        available_df = pd.DataFrame([( ' '.join(i.split('_')[:-1]),int(i.split('_')[-1].rstrip('.jpg')),i) 
              for i in os.listdir(IMAGE_DIR) if i.endswith('.jpg')],columns=['name','image_id','image_path'])
        df = available_df.merge(df[['name','image_id','bbox']],on=['name','image_id'],how='inner').drop_duplicates('image_path')

        print(f'There are {len(df)} images available for evaluation.'.format(len(df)))
    except FileNotFoundError:
        print(f"Error: Metadata file '{CSV_FILE}' not found.")
        return

    genuine_pairs, imposter_pairs = generate_evaluation_pairs(df, IMAGE_DIR, NUM_PAIRS)

    if not genuine_pairs and not imposter_pairs:
        print("Could not generate any pairs for evaluation. Exiting...")
        return

    distances = []
    labels = [] # 1 for genuine, 0 for imposter

    print(f"\nCalculating distances for {NUM_PAIRS} genuine pairs...")
    for _, ((path1, bbox1_str), (path2, bbox2_str)) in tqdm(enumerate(genuine_pairs),total=NUM_PAIRS):
        emb1 = get_embedding(path1, cu.parse_bbox(bbox1_str))
        emb2 = get_embedding(path2, cu.parse_bbox(bbox2_str))
        if emb1 is not None and emb2 is not None:
            dist = np.linalg.norm(emb1 - emb2)
            distances.append(dist)
            labels.append(1)

    print(f"\nCalculating distances for {NUM_PAIRS} imposter pairs...")
    for _, ((path1, bbox1_str), (path2, bbox2_str)) in tqdm(enumerate(imposter_pairs),total=NUM_PAIRS):
        emb1 = get_embedding(path1, cu.parse_bbox(bbox1_str))
        emb2 = get_embedding(path2, cu.parse_bbox(bbox2_str))
        if emb1 is not None and emb2 is not None:
            dist = np.linalg.norm(emb1 - emb2)
            distances.append(dist)
            labels.append(0)
        
    distances = np.array(distances)
    labels = np.array(labels)

    if len(distances) == 0:
        print("No valid distances were calculated. Cannot evaluate.")
        return

    # --- Calculate Metrics ---
    thresholds = np.arange(0, 2, 0.01)
    accuracies = []
    best_accuracy = 0
    best_threshold = 0

    for threshold in tqdm(thresholds,desc='Checking thresholds'):
        predictions = (distances <= threshold).astype(int) # Lower distances means more similar
        accuracy = np.mean(predictions == labels)
        accuracies.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            
    print("\n--- Evaluation Results ---")
    print(f"Best Accuracy: {best_accuracy:.4f} at Threshold: {best_threshold:.2f}")
    print("This threshold value is a good candidate for DUPLICATE_THRESHOLD in app.py.")

    # --- Plotting ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, accuracies)
    plt.title('Accuracy vs. Distance Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Best Threshold ({best_threshold:.2f})')
    plt.legend()

    fpr, tpr, _ = roc_curve(labels, -distances) 
    roc_auc = auc(fpr, tpr)

    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("Model_eval.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
