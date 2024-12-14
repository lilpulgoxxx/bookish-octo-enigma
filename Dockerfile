# Use a base Python image with specific version
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for building libraries and dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libsndfile1 \
    ffmpeg \
    git \
    wget \
    curl \
    espeak-ng \
    espeak \
    ffmpeg \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files from your project into the container
COPY . .

# Expose the port the app runs on
EXPOSE 7860
