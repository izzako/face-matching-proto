Face Matching MVP for Duplicate Account Detection
-------------------------------------------------
This project is a Minimum Viable Product (MVP) for a system that detects duplicate user accounts by matching faces from selfies against a database of existing users.

It uses the FaceNet (InceptionResNetV1) for the image embbedings, Qdrant client as vector database, Gradio for the web interface, and Docker for the containerization.

# Project Structure
```
.
├── a_images/              # Directory for downloaded face images
├── b_database/            # Directory for the vector database and metadata
├── app.py                 # The main Gradio web application
├── build_database.py      # Script to create face embeddings and the populate the database
├── config.yaml            # Generall app config file
├── image_downloader.py    # Script to download images from the CSV
├── model_evaluation.py    # Script to evaluate model and choose threshold
├── facescrub_metadata.csv # The original metadata file (provided)
├── Dockerfile             # Docker configuration for deployment
└── requirements.txt       # Python dependencies
```

# How to Run
There are two ways to run this application: locally with Python or using Docker.

Make sure you have Python 3.11 or higher installed. I also recommend checking out `config.yaml` file for the default configurations.

## 1. Running Locally

### Step 1: Setup Environment
It is recommended to use a virtual environment.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You may need to install `cmake` and `C++` build tools first:
`sudo apt-get update && sudo apt-get install build-essential cmake`


### Step 2: Download the Data
Run the download script. This will populate the `a_images` directory. It might take a while and you will see some errors for broken URLs, which is expected.
```
python image_downloader.py
```

### Step 2.5 (Optional): Tuning the Threshold
Run the model evaluation script. It will create pairs of images of the same person and different person. The number of pairs `EVAL_NUM_PAIRS` is set in `config.yaml`. The script will output `Model_Eval.png` and recommend a threshold value based on best accuracy **(i personally DID NOT recommend any number below 0.5)**.
```
python model_evaluation.py
```

### Step 3: Build the Vector Database
This script processes the images, creates face embeddings, and saves it to Qdrant client in the `b_database` directory.

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
### Step 2: Create and Run the Docker Container
This command create the container named face-matching and runs the app and maps the container's port 7860 to our local machine's port 7860, while also retaining the `config.yaml` file in the container (no need to rebuild the image).
```
docker run -d\
    --name face-matching\
    -p 7860:7860 \
    -v $(pwd)/config.yaml:/app/config.yaml \
    face-matching-app
```

Open your browser and navigate to `http://127.0.0.1:7860`.

to start the container, you can use the following command:
```
docker start -ai face-matching 
```
and to stop the container, you can use the following command:
```
docker stop face-matching
``` 

