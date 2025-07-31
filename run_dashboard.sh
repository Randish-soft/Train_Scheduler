#!/bin/bash

# Railway Dashboard Launcher Script

echo "🚄 Railway Analysis Dashboard Launcher"
echo "===================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Unix/MacOS
    source venv/bin/activate
fi

# Install requirements
echo "Installing dependencies..."
pip install -r dashboard_requirements.txt

# Launch dashboard
echo "Starting Railway Dashboard..."
echo "===================================="
echo "Dashboard will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the dashboard"
echo ""

streamlit run railway_dashboard.py