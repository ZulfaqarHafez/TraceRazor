"""Shared fixtures for the TraceRazor v2 test suite."""
import sys
from pathlib import Path

# Make the package importable without installing it
sys.path.insert(0, str(Path(__file__).parent.parent))
