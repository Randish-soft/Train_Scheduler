# ----- base image ---------------------------------------------------------
    FROM mcr.microsoft.com/vscode/devcontainers/python:0-3.11

    # ----- OS libraries you might need ----------------------------------------
    RUN apt-get update && \
        apt-get install -y --no-install-recommends \
            build-essential graphviz libgraphviz-dev && \
        rm -rf /var/lib/apt/lists/*
    
    # ----- Python deps --------------------------------------------------------
    COPY Model/requirements.txt /tmp/
    RUN pip install --no-cache-dir -r /tmp/requirements.txt \
        dagster dagster-webserver ipympl jupyterlab folium
    
    # pre-install JupyterLab extensions that VS Code webview understands
    RUN jupyter labextension install @jupyterlab/server-proxy --no-build
    
    # ----- workspace ----------------------------------------------------------
    WORKDIR /workspace
    
    # VS Code tasks (“Run Dagster”, “Run Jupyter”) will cd here
    