# ────────────────────────────────────────────────────────────────
# Base
FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.11

# ────────────────────────────────────────────────────────────────
# OS libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential graphviz libgraphviz-dev && \
    rm -rf /var/lib/apt/lists/*

# ────────────────────────────────────────────────────────────────
# Python deps
COPY Model/requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt \
    dagster dagster-webserver

# ────────────────────────────────────────────────────────────────
WORKDIR /workspace
