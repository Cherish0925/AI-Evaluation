# /// script
# requires-python = ">=3.11"
# dependencies = ["mcp>=1.29,<2"]
# ///
"""Relocatable entry point for source checkouts and agent plugins."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ai_eval.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
