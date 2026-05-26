import runpy, sys
from pathlib import Path

# Add repo root to path so bfmix package is importable
sys.path.insert(0, str(Path(__file__).parent))

# Run the actual Streamlit app
runpy.run_path(str(Path(__file__).parent / "webapp" / "app.py"), run_name="__main__")

