# Stage 1: Build stage to handle dependencies
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the requirements file first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Final application stage
FROM python:3.11-slim

WORKDIR /app

# Copy the installed packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy the application source code
COPY . .

# --- Data Preparation Steps ---
# These commands run ONCE during the `docker build` process.
# This pre-builds the database so the container starts up ready to go.

# 1. Download images from the provided CSV
RUN python image_downloader.py

# 2. Build the vector database from the downloaded images
RUN python build_database.py

# Expose the port Gradio will run on
EXPOSE 7860

# Command to run the application when the container starts
CMD ["python", "app.py"]
