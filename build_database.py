import numpy as np
import os
import pandas as pd
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



def main():
    #refresh_new
    if os.path.isfile(LOG_FILE):
        os.remove(LOG_FILE)
        
    # Initialize Qdrant client. This will create a local, file-based DB.
    if not os.path.exists(DATABASE_DIR):
        os.makedirs(DATABASE_DIR)
        s_log(f"Created directory: {DATABASE_DIR}")

    client = QdrantClient(path=QDRANT_PATH)

    # Get the embedding dimension from the model
    embedding_dim = cu.config['MODEL']['EMBEDDING_DIM']

    # Create a new collection in Qdrant if it doesn't exist
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=embedding_dim, distance=models.Distance.COSINE),
        )
        s_log(f"Qdrant collection '{COLLECTION_NAME}' created/recreated successfully.")

    try:
        df = pd.read_csv(CSV_FILE)

        # Only select available (downloaded) images
        df['filename'] = df['name'].str.replace(' ','_') + '_' + df['image_id'].astype(str) + '_' + df['face_id'].astype(str) + '.jpg'
        df = df[df['filename'].isin([i for i in os.listdir(IMAGE_DIR) if i.endswith('.jpg')])].copy()
        df['image_dir'] = IMAGE_DIR

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

        bbox = cu.parse_bbox(bbox_str)

        embedding,error_msg = cu.get_embedding(image_path, bbox)
        
        if embedding is None:
            s_log(error_msg)
        else:
        # Prepare the point for Qdrant
            points_to_upsert.append(
                models.PointStruct(
                    id=str(uuid4()), # Each point needs a unique ID
                    vector=embedding.tolist(),
                    payload=row[['image_dir','filename']].to_dict() # Store metadata here
                )
            )

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