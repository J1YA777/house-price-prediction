import os
from pathlib import Path
import joblib

def save_artifact(obj, path):
    """Save any Python object (model, encoder, preprocessor) to a file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_artifact(path):
    """Load a saved Python object."""
    return joblib.load(path)

def ensure_dir(path):
    """Ensure a directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

