import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
timesformer_root = project_root / 'external' / 'TimeSformer'
# sys.path.insert(0, str(timesformer_root))
# sys.path.insert(0, str(project_root))