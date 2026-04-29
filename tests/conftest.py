# tests/conftest.py
import sys
from unittest.mock import MagicMock

sys.modules["litert_lm"] = MagicMock()
