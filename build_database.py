import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
import numpy as np
import os
from PIL import Image
import pandas as pd
from torchvision import transforms
from tqdm import tqdm
from qdrant_client import QdrantClient, models
from uuid import uuid4
import custom_utils as cu
from functools import partial

# --- Configuration ---
IMAGE_DIR = cu.config['DIRECTORIES']['IMAGE_DIR']
DATABASE_DIR = cu.config['DIRECTORIES']['DATABASE_DIR']
CSV_FILE = cu.config['DIRECTORIES']['CSV_FILE']
QDRANT_PATH = os.path.join(DATABASE_DIR, cu.config['DATABASE']['QDRANT_PATH'])
COLLECTION_NAME = cu.config['DATABASE']['COLLECTION_NAME']
LOG_FILE = os.path.join(cu.config['DIRECTORIES']['LOG_DIR'],'build_database.log')

# Write log into log file 
s_log = partial(cu.simple_log, LOG_FILE)

# --- Model & Device Setup ---
device = torch.device(cu.config['MODEL']['device'])
resnet = InceptionResnetV1(pretrained='vggface2', device=device).eval()
s_log("FaceNet model loaded successfully on CPU.")



def main():
    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)
        s_log(f"Created directory: {DATABASE_DIR}")

    if os.path.isfile(LOG_FILE):
        os.remove(LOG_FILE)
    # Initialize Qdrant client. This will create a local, file-based DB.
    client = QdrantClient(path=QDRANT_PATH)

    # Get the embedding dimension from the model
    embedding_dim = cu.config['MODEL']['EMBEDDING_DIM']

    # Create a new collection in Qdrant if it doesn't exist
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.EUCLID),
        )
        s_log(f"Qdrant collection '{COLLECTION_NAME}' created/recreated successfully.")

    try:
        df = pd.read_csv(CSV_FILE)

        # Only select available (downloaded) images
        available_df = pd.DataFrame([( ' '.join(i.split('_')[:-1]),int(i.split('_')[-1].rstrip('.jpg')),i) 
                for i in os.listdir(IMAGE_DIR) if i.endswith('.jpg')],columns=['name','image_id','filename'])
        df = available_df.merge(df[['name','image_id','bbox']],on=['name','image_id'],how='inner').drop_duplicates('filename')
    except FileNotFoundError:
        s_log(f"ERROR: The file '{CSV_FILE}' was not found.")
        return

    points_to_upsert = []
    s_log(f"Found {len(df)} images in '{IMAGE_DIR}'. Starting embedding generation...")


    for _, row in tqdm(df.iterrows(),total=len(df),desc='Generating image embeddings'):
        image_path = os.path.join(IMAGE_DIR, row['filename'])
        bbox_str = row['bbox']
        
        if bbox_str is None:
            continue

        try:
            img = Image.open(image_path).convert('RGB')
            bbox = cu.parse_bbox(bbox_str)
            if bbox is None:
                continue
            
            face_img = img.crop(bbox)
            face_tensor = cu.preprocess(face_img).to(device)
            
            with torch.no_grad():
                embedding = resnet(face_tensor.unsqueeze(0)).detach().cpu().numpy().flatten()
                embedding = cu.normalize_embedding(embedding)
            
            # Prepare the point for Qdrant
            points_to_upsert.append(
                models.PointStruct(
                    id=str(uuid4()), # Each point needs a unique ID
                    vector=embedding.tolist(),
                    payload=row[['name','image_id','filename']].to_dict() # Store metadata here
                )
            )
            
        except Exception as e:
            s_log(f"Error processing {image_path}: {e}")

    if not points_to_upsert:
        s_log("No embeddings were generated. Exiting.")
        return

    # Upsert all points to Qdrant in batches for efficiency
    s_log(f"\nUpserting {len(points_to_upsert)} points to Qdrant...")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points_to_upsert,
        wait=True # Wait for the operation to complete
    )
        
    s_log(f"Qdrant database built successfully at: {QDRANT_PATH}")
    s_log("\n--- Database Build Complete ---")

if __name__ == "__main__":
    main()