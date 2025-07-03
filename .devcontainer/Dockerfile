ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim

ENV POETRY_HOME=/opt/poetry \
    POETRY_VIRTUALENVS_CREATE=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential git curl gdal-bin libgdal-dev libproj-dev proj-data proj-bin \
        libspatialindex-dev graphviz && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 - && \
    ln -s $POETRY_HOME/bin/poetry /usr/local/bin/poetry

WORKDIR /workspace

# ----- deps -------
COPY pyproject.toml ./
# COPY poetry.lock ./        # ← uncomment if you have it
RUN poetry install --no-interaction --no-ansi --no-root

# Dagster UI + daemon client
RUN pip install dagster dagster-webserver

# ----- copy code last ----
COPY . .

# Default command is just a shell; VS Code will run postStartCommand
CMD ["bash"]
