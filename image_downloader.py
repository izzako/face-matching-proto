import pandas as pd
import requests
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import custom_utils as cu

# --- Configuration ---
CSV_FILE = cu.config['DIRECTORIES']['CSV_FILE']
IMAGE_DIR = cu.config['DIRECTORIES']['IMAGE_DIR']
MAX_WORKERS = cu.config['DOWNLOADER']['MAX_WORKERS']
LOG_FILE = os.path.join(cu.config['DIRECTORIES']['LOG_DIR'],'download_image.log')

# Write log into log file
s_log = partial(cu.simple_log, LOG_FILE)

# --- Main Logic ---
def download_image(args):
    """
    Downloads a single image from a URL and saves it.
    Handles potential errors during download.
    """
    index, row = args
    name = row['name'].replace(' ', '_')
    image_id = row['image_id']
    face_id = row['face_id']
    url = row['url']
    filename = os.path.join(IMAGE_DIR, f"{name}_{image_id}_{face_id}.jpg")

    # Skip if the file already exists
    if os.path.exists(filename):
        s_log(f"[{index}] Skipping, already exists: {filename}")
        
        return None

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        with open(filename, 'wb') as f:
            f.write(response.content)
        s_log(f"[{index}] Successfully downloaded: {filename}")
        return filename
    except requests.exceptions.RequestException as e:
        s_log(f"[{index}] Failed to download {url}: {e}\n")
        return None

def clean_broken_images(folder, delete=False):
    """
    Scan a folder and detect/remove broken images.

    Parameters:
        folder (str or Path): Path to the folder containing images.
        delete (bool): If True, delete broken images. Otherwise, just log them.
    """
    folder = Path(folder)
    broken = []

    for file in tqdm(folder.glob("*"),desc='Verifying images integrity'):
        if file.is_file():
            try:
                with Image.open(file) as img:
                    img.verify()  # verify doesn't load full image, just checks integrity
            except (UnidentifiedImageError, OSError):
                broken.append(file)
                if delete:
                    try:
                        file.unlink()
                        s_log(f"Deleted broken image: {file}")
                    except Exception as e:
                        s_log(f"Failed to delete {file}: {e}")
                else:
                    s_log(f"Broken image detected: {file}")

    if delete:
        s_log(f"\nScan complete. Deleted {len(broken)} broken images.")
    else:
        s_log(f"\nScan complete. Found {len(broken)} broken images.")
    return broken

def main():
    """
    Main function to orchestrate the image download process.
    """
    # 1. Ensure the output directory exists
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
        s_log(f"Created directory: {IMAGE_DIR}")

    if os.path.isfile(LOG_FILE):
        os.remove(LOG_FILE)

    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    # 2. Read the metadata
    try:
        df = pd.read_csv(CSV_FILE)
    except FileNotFoundError:
        s_log(f"Error: The file '{CSV_FILE}' was not found. Please make sure it's in the same directory.")
        return

    # 3. Create a list of tasks for the thread pool
    tasks = [(index, row) for index, row in df.iterrows()]

    # 4. Use a ThreadPoolExecutor for parallel downloads
    s_log(f"Starting download of {len(tasks)} images with {MAX_WORKERS} workers...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(tqdm(executor.map(download_image, tasks),total=len(tasks),desc='Downloading Images'))

    successful_downloads = [res for res in results if res is not None]
    s_log("\n--- Download Complete ---")
    s_log(f"Successfully downloaded {len(successful_downloads)} / {len(tasks)} images.")
    
    # 5. Delete downloaded but broken images (cannot be opened by pillow)
    clean_broken_images(IMAGE_DIR, delete=True)

if __name__ == "__main__":
    main()
