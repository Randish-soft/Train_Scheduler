import numpy
import pandas
import csv
import time
from pathlib import Path
from typing import Optional, List
import logging
import argparse
import sys

def Welcome_Message():
    banner = """
    ╔════════════════════════════════════════════════════════════╗
    ║                    🚄 RAILWAY Raster 🚄                    ║
    ║              Intelligent Route Planning System             ║
    ║                                                            ║
    ║  Learn → Generate → Optimize → Validate → Deploy           ║
    ╚════════════════════════════════════════════════════════════╝
    What Does this System Do?
    - Estimate Costs
    - Choose Best routes for Budget and Demand
    - Choose Best Train(s) for route(s)
    - Estimate Building Time
    - Estimate Number of Jobs Created
    """
    print(banner)
    
def Helper():
    text = """
    ---!!RUN FROM ROOT!!---
    
    Running pipeline:  poetry 
    
    With Logger: 
    """