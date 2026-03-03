import json
import os
import datetime
from pathlib import Path

METADATA_PATH = Path("models/metadata.json")

def update_model_metadata(version: str, metrics: dict, model_path: str = "models/user_tower_quantized.onnx"):
    """
    Saves model performance stats to a JSON registry.
    This file is read by the FastAPI /models/current endpoint.
    """
    metadata = {
        "model_version": version,
        "last_updated": datetime.datetime.now().isoformat(),
        "framework": "ONNX (Quantized)",
        "performance": {
            "auc": metrics.get("auc", 0.0),
            "mae_parity": metrics.get("mae", 0.0),
            "mean_latency_ms": metrics.get("onnx_mean_ms", 0.0),
            "p99_latency_ms": metrics.get("onnx_p99_ms", 0.0)
        },
        "artifacts": {
            "onnx_model": model_path,
            "registry_type": "mlflow-linked-registry"
        }
    }

    os.makedirs(METADATA_PATH.parent, exist_ok=True)

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    
    print(f"✅ Metadata successfully generated at {METADATA_PATH}")

def get_current_metadata():
    """Reads the metadata.json file for the API and Dashboard."""
    if not METADATA_PATH.exists():
        return {
            "model_version": "v0.0.0-base",
            "performance": {"auc": 0.0},
            "status": "No production metadata found."
        }
    
    with open(METADATA_PATH, "r") as f:
        return json.load(f)