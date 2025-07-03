FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_HOME="/opt/poetry"

# ---------- system deps ----------
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
        build-essential \
        gdal-bin libgdal-dev \
        libproj-dev proj-data proj-bin \
        libspatialindex-dev \
        git curl && \
    rm -rf /var/lib/apt/lists/*

# ---------- Poetry ----------
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s $POETRY_HOME/bin/poetry /usr/local/bin/poetry

WORKDIR /workspace

# ---------- dependency installation ----------
COPY pyproject.toml poetry.lock* ./               # lock file is optional
RUN poetry config virtualenvs.create false && \
    poetry install --no-dev --no-interaction --no-ansi

# ---------- project code ----------
COPY . .

CMD ["python", "-m", "src"]   # falls back to src/__main__.py if provided
