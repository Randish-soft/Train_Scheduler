# .devcontainer/Dockerfile -----------------------------------------------
    FROM mambaorg/micromamba:1.5.7-base   #  ↳ already has a conda “base” env

    ENV MAMBA_DOCKERFILE_ACTIVATE=1 \
        CONDA_PKGS_DIRS=/opt/conda/pkgs \
        POETRY_HOME=/opt/poetry \
        POETRY_VIRTUALENVS_CREATE=false \
        PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
    
    # ------------------------------------------------------------------ deps --
    RUN micromamba install -y -n base -c conda-forge \
          python=3.10 richdem=0.3.4 gdal proj geos libspatialindex graphviz \
     && micromamba clean -ay
    
    # ---------------------------------------------------------------- Poetry --
    RUN curl -sSL https://install.python-poetry.org | python - \
     && ln -s $POETRY_HOME/bin/poetry /usr/local/bin/poetry
    
    WORKDIR /workspace
    COPY pyproject.toml poetry.lock* ./
    RUN poetry install -E terrain --no-interaction --no-ansi
    
    # copy code last
    COPY . .
    
    EXPOSE 3000 8888
    CMD ["bash"]
    