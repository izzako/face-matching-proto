
import numpy as np
import pandas as pd
import os
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

def generate_evaluation_pairs(df, image_dir, num_pairs):
    """Generates balanced genuine and imposter pairs based on available images."""

    bbox_lookup = df.set_index('filename')['bbox'].to_dict()

    # 1. Scan available images and group them by person
    person_to_images = {}
    for _,row in df.iterrows():
        try:
            # Filename is like 'Name_Lastname_imageid_face_id.jpg'
            name = row['name']
            if name not in person_to_images:
                person_to_images[name] = []
            person_to_images[name].append(row['filename'])
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
        
        bbox1_str = bbox_lookup.get(file1)
        bbox2_str = bbox_lookup.get(file2)
        
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
        
        bbox1_str = bbox_lookup.get(file1)
        bbox2_str = bbox_lookup.get(file2)
        
        if bbox1_str and bbox2_str:
            imposter_pairs.append(((path1, bbox1_str), (path2, bbox2_str)))
            
    return genuine_pairs, imposter_pairs

# --- Main Evaluation Logic ---
def main():
    try:
        df = pd.read_csv(CSV_FILE)
        df['filename'] = df['name'].str.replace(' ','_') + '_' + df['image_id'].astype(str) + '_' + df['face_id'].astype(str) + '.jpg'
        # filter to available image only
        df = df[df['filename'].isin([i for i in os.listdir(IMAGE_DIR) if i.endswith('.jpg')])].copy()

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
        emb1,_ = cu.get_embedding(path1, cu.parse_bbox(bbox1_str))
        emb2,_ = cu.get_embedding(path2, cu.parse_bbox(bbox2_str))
        if emb1 is not None and emb2 is not None:
            dist = cu.cosine_similarity(emb1, emb2)
            distances.append(dist)
            labels.append(1)

    print(f"\nCalculating distances for {NUM_PAIRS} imposter pairs...")
    for _, ((path1, bbox1_str), (path2, bbox2_str)) in tqdm(enumerate(imposter_pairs),total=NUM_PAIRS):
        emb1,_ = cu.get_embedding(path1, cu.parse_bbox(bbox1_str))
        emb2,_ = cu.get_embedding(path2, cu.parse_bbox(bbox2_str))
        if emb1 is not None and emb2 is not None:
            dist = cu.cosine_similarity(emb1,emb2)
            distances.append(dist)
            labels.append(0)
        
    distances = np.array(distances)
    labels = np.array(labels)

    if len(distances) == 0:
        print("No valid distances were calculated. Cannot evaluate.")
        return

    # --- Calculate Metrics ---
    thresholds = np.arange(-1, 1, 0.05)
    accuracies = []
    best_accuracy = 0
    best_threshold = 0

    for threshold in tqdm(thresholds,desc='Checking thresholds'):
        predictions = (distances >= threshold).astype(int) # higher distance means more similar
        accuracy = np.mean(predictions == labels)
        accuracies.append(accuracy)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            
    print("\n--- Evaluation Results ---")
    print(f"Best Accuracy: {best_accuracy:.4f} at Threshold: {best_threshold:.2f}")
    print("This threshold value is a good candidate for DUPLICATE_THRESHOLD in config.yaml")

    # --- Plotting ---
    plt.figure(figsize=(18, 5))

    # 1. Accuracy vs Threshold
    plt.subplot(1, 3, 2)
    plt.plot(thresholds, accuracies)
    plt.title('Accuracy vs. Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.axvline(x=best_threshold, color='r', linestyle='--', label=f'Threshold ({best_threshold:.2f})')
    plt.legend()

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(labels, distances) 
    roc_auc = auc(fpr, tpr)

    plt.subplot(1, 3, 3)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)

    # 3. Histogram of similarities
    plt.subplot(1, 3, 1)
    genuine = distances[labels == 1]
    impostor = distances[labels == 0]
    plt.hist(genuine, bins=50, alpha=0.6, label="Genuine (same person)")
    plt.hist(impostor, bins=50, alpha=0.6, label="Impostor (different person)")
    plt.axvline(best_threshold, color='red', linestyle='--', label=f'Threshold {best_threshold:.2f}')
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Count")
    plt.title("Distribution of Similarities")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("Model_eval.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    main()
