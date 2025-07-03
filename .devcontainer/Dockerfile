# ---------- base image tag ----------
    ARG PYTHON_VERSION=3.11              # must be declared before FROM
    FROM python:${PYTHON_VERSION}-slim
    
    # ---------- env + poetry ----------
    ENV POETRY_HOME=/opt/poetry \
        POETRY_VIRTUALENVS_CREATE=false \
        PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1
    
    # ---------- system deps ----------
    RUN apt-get update -qq && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            build-essential git curl gdal-bin libgdal-dev libproj-dev proj-data proj-bin \
            libspatialindex-dev graphviz && \
        rm -rf /var/lib/apt/lists/*
    
    # ---------- poetry ----------
    RUN curl -sSL https://install.python-poetry.org | python3 - && \
        ln -s $POETRY_HOME/bin/poetry /usr/local/bin/poetry
    
    WORKDIR /workspace
    
# ---------- project deps ----------
    COPY pyproject.toml poetry.lock* README.md ./   # <─ add README.md (match the exact case!)
    RUN poetry install --no-interaction --no-ansi   # still installs your package
    
    
    # ---------- dagster + dagit ----------
    RUN pip install dagster dagit
    
    # ---------- copy source last ----------
    COPY . .
    
    CMD ["dagit", "-h", "0.0.0.0", "-p", "3000"]
    