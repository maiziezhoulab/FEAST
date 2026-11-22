"""Place panels on the existing journal page size."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from feast_plate import assemble_main
from layout import FIGURE, PLATE, PAGE, FRAMES


if __name__ == "__main__":
    assemble_main(FIGURE, PLATE, PAGE, FRAMES)
