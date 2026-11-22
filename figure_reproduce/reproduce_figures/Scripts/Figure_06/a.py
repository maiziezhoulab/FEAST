"""Render Figure_06 panel a. """
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from panels import main


if __name__ == "__main__":
    main('a')
