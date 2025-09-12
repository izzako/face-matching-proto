import torch
from facenet_pytorch import MTCNN
import numpy as np
import os
from PIL import Image
import gradio as gr
from qdrant_client import QdrantClient, models
from uuid import uuid4
import custom_utils as cu

# --- Configuration ---
DATABASE_DIR = cu.config['DIRECTORIES']['DATABASE_DIR']
UPLOAD_DIR = cu.config['DIRECTORIES']['UPLOAD_DIR']
IMAGE_DIR = cu.config['DIRECTORIES']['IMAGE_DIR']
QDRANT_PATH = os.path.join(DATABASE_DIR, cu.config['DATABASE']['QDRANT_PATH'])
COLLECTION_NAME =cu.config['DATABASE']['COLLECTION_NAME']
DUPLICATE_THRESHOLD = cu.config['APP']['DUPLICATE_THRESHOLD'] # L2 distance threshold for FaceNet. Found from evaluation.

# --- Model & Device Setup ---
device = torch.device(cu.config['MODEL']['device'])
# MTCNN is needed here to detect the face in the newly uploaded selfie
mtcnn = MTCNN(keep_all=False, device=device) 
resnet = cu.resnet
print(f"Models loaded successfully on {device.type.upper()}.")

# --- Qdrant Client Setup ---
# Connect to the local Qdrant DB file
client = QdrantClient(path=QDRANT_PATH)
print(f"Connected to Qdrant DB at: {QDRANT_PATH}")


# Ensure upload directory exists
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Core Application Logic ---
def get_embedding_from_upload(image):
    """Detects a face in an uploaded image and returns its embedding."""
    try:
        # MTCNN returns a 160x160 cropped face tensor, ready for the model
        face_tensor = mtcnn(image)
        if face_tensor is None:
            return None, "No face detected in the uploaded image."
            
        face_tensor = face_tensor.to(device)
        
        with torch.no_grad():
            embedding = resnet(face_tensor.unsqueeze(0))
            
        return embedding.detach().cpu().numpy().flatten(), None
    except Exception as e:
        return None, f"An error occurred during face detection: {e}"
    
def find_matches(uploaded_image):
    """
    Main function to handle image upload, find matches in Qdrant,
    and update the database if no duplicate is found.
    """
    if uploaded_image is None:
        return "No duplicate found.", "Please upload an image.", None, [], []

    # 1. Generate embedding for the uploaded image
    query_embedding, error = get_embedding_from_upload(uploaded_image)
    if error:
        return "Error", error, None, [], []
        
    # 2. Search for the closest matches in Qdrant
    search_result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding.tolist(),
        limit=6 # Retrieve top 6 matches
    ).points

    # 3. Process search results
    best_match = search_result[0] if search_result else None
    
    # Prepare gallery output
    match_images = [os.path.join(hit.payload['image_dir'],hit.payload['filename']) for hit in search_result]
    match_scores = [f"Score: {hit.score:.4f}" for hit in search_result]
    output_match = [(image,score) for image,score in zip(match_images,match_scores)]

    if best_match and best_match.score >= DUPLICATE_THRESHOLD:
        # Duplicate Detected
        flag = "Duplicate Detected"
        best_score = best_match.score
        message = f"Closest match found with a similarity score of {best_score:.4f}."
        
        
        
    else:
        # No Duplicate Found, add to DB
        flag = "No Duplicate Found"
        if best_match:
             best_score = best_match.score
             message = f"Closest match is at similarity score of {best_score:.4f}, which is below the threshold of {DUPLICATE_THRESHOLD}."
        else:
            message = "Database was empty."
        match_images = []
        match_scores = []
        # Save the new user's image

        # new_user_id = str(uuid4()) #assign random id
        # new_image_filename = f"new_user_{new_user_id}.jpg"
        # uploaded_image.save( os.path.join(UPLOAD_DIR, new_image_filename))
        
        # Add the new embedding to the Qdrant collection

        # client.upsert(
        #     collection_name=COLLECTION_NAME,
        #     points=[
        #         models.PointStruct(
        #             id=new_user_id,
        #             vector=query_embedding.tolist(),
        #             payload={'image_dir':UPLOAD_DIR,
        #                      "filename": new_image_filename}
        #         )
        #     ],
        #     wait=True
        # )

    return flag, flag + ": " + message, f"{best_score:.4f}" if best_score else "N/A", output_match

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Duplicate Face Detection")
    gr.Markdown("Upload a photo to check if a similar face already exists in the database.")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Upload Photo")
            submit_btn = gr.Button("Check for Duplicates", variant="primary")
            
        with gr.Column(scale=2):
            flag_output = gr.Label(label="Result")
            score_output = gr.Textbox(label="Best Similarity Score (Cosine Similarity)", interactive=False)
            message_output = gr.Textbox(label="Details", interactive=False)
    
    gr.Markdown("### Closest Matches from Database")
    gallery_output = gr.Gallery(label="Matching Images", show_label=False, elem_id="gallery", columns=6, object_fit="contain", height="auto")
    
    submit_btn.click(
        fn=find_matches,
        inputs=image_input,
        outputs=[flag_output, message_output, score_output, gallery_output]
    )


if __name__ == "__main__":
    demo.launch()
