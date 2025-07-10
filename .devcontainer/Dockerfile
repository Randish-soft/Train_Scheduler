# Use a slim Python image
FROM python:3.11-slim-bullseye

# Avoid interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl build-essential git \
    gdal-bin libgdal-dev python3-dev \
    && apt-get clean

# Set working directory
WORKDIR /workspace

# ✅ Copy requirements.txt from inside .devcontainer
COPY --chown=railway:railway .devcontainer/requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
