"""
Processing Input: Getting country naem and getitng cities and data (everyhthing)
"""

import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
import requests
from collections import defaultdict
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

