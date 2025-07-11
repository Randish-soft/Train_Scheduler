FROM mcr.microsoft.com/devcontainers/python:3.13

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libproj-dev \
    libgeos-dev \
    libspatialindex-dev \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for GDAL
ENV GDAL_CONFIG=/usr/bin/gdal-config
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

# Install Poetry
RUN pip install poetry

# Configure Poetry
RUN poetry config virtualenvs.create false

# Set working directory
WORKDIR /workspace

# Install Python dependencies directly with pip (simpler approach)
RUN pip install \
    pandas>=2.0.0 \
    numpy>=1.24.0 \
    geopandas>=0.14.0 \
    shapely>=2.0.0 \
    rasterio>=1.3.0 \
    requests>=2.31.0 \
    geopy>=2.3.0 \
    matplotlib>=3.7.0 \
    scipy>=1.11.0 \
    scikit-learn>=1.3.0 \
    pyproj>=3.6.0 \
    dagster>=1.5.0 \
    dagster-webserver>=1.5.0 \
    jupyter

# Expose ports for Dagster and Jupyter
EXPOSE 3000 8080 8888

# Create cache directories
RUN mkdir -p data/_cache/terrain

# Set default command
CMD ["bash"]