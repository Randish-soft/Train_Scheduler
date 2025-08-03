"""
Route Constraint Analyzer
=========================

Analyzes routing constraints including:
- NIMBY factors (residential impacts, community concerns)
- Environmental constraints (protected areas, wetlands)
- Infrastructure conflicts (roads, utilities, airports)
- Economic factors (land costs, construction complexity)
- Regulatory requirements (permits, approvals)

This module works with route data from plotting_route.py

Author: Miguel Ibrahim E
"""

import logging
import math
from typing import Dict, List, Any, Tuple
from enum import Enum
from pathlib import Path
import json
