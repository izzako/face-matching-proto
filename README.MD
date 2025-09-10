Face Matching MVP for Duplicate Account Detection
-------------------------------------------------
This project is a Minimum Viable Product (MVP) for a system that detects duplicate user accounts by matching faces from selfies against a database of existing users.

It uses the FaceNet (InceptionResNetV1) for the image embbedings, FAISS for efficient similarity search in a vector database, Gradio for the web interface, and Docker for containerization.

# Project Structure
```
.
├── a_images/              # Directory for downloaded face images
├── b_database/            # Directory for the FAISS index and metadata
├── app.py                 # The main Gradio web application
├── build_database.py      # Script to create face embeddings and the FAISS index
├── image_downloader.py     # Script to download images from the CSV
├── model_evaluation.py    # Script to evaluate choosen model
├── facescrub_metadata.csv # The original metadata file (provided)
├── Dockerfile             # Docker configuration for deployment
└── requirements.txt       # Python dependencies
```

# How to Run
There are two ways to run this application: locally with Python or using Docker.

## 1. Running Locally

### Step 1: Setup Environment
It is recommended to use a virtual environment.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You may need to install `cmake` and C++ build tools first:
`sudo apt-get update && sudo apt-get install build-essential cmake`

### Step 2: Download the Data
Run the download script. This will populate the `a_images` directory. It might take a while and you will see some errors for broken URLs, which is expected.
```
python image_downloader.py
```
### Step 3: Build the Vector Database
This script processes the images, creates face embeddings, and saves a FAISS index in the `b_database` directory.

```
python build_database.py
```

### Step 4: Run the Gradio Web App
```
python app.py
```

Open your browser and navigate to `http://127.0.0.1:7860`.

## 2. Running with Docker

### Step 1: Build the Docker Image
Make sure you have Docker installed. The following command builds the image, which includes running the download and database-building steps inside the container.
```
docker build -t face-matching-app .
```
### Step 2: Run the Docker Container
This command runs the app and maps the container's port 7860 to your local machine's port 7860.
```
docker run -p 7860:7860 face-matching-app
```

Open your browser and navigate to `http://127.0.0.1:7860`.

