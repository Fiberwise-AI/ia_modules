"""
Test configuration and fixtures
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# This file can be used to define pytest fixtures that are shared across all tests

# Configure asyncio for pytest
pytest_plugins = [
    "pytest_asyncio",
]

# Set up event loop policy for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
