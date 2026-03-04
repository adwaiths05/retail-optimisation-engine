import sys
import shutil
import os
import datetime
import json
from training.benchmark import benchmark  # Existing benchmark script
from mlops.model_registry import update_model_metadata

# CONFIGURATION
# Tightened threshold to 1ms based on ONNX benchmarks (~0.05ms)
LATENCY_THRESHOLD_MS = 2.0  
MAE_THRESHOLD = 0.01          # Accuracy Parity Budget
SOURCE_MODEL = "models/user_tower_quantized.onnx"
PROD_MODEL = "models/user_tower_production.onnx"
MAPPINGS_FILE = "models/mappings.pkl"
ARCHIVE_DIR = "models/archive"

def publish_production_model():
    print("🚀 Starting Production Promotion Pipeline...")

    # 1. Check if required model and artifact files exist
    if not os.path.exists(SOURCE_MODEL):
        print(f"❌ Error: {SOURCE_MODEL} not found. Did training/quantization finish?")
        sys.exit(1)
    
    if not os.path.exists(MAPPINGS_FILE):
        print(f"❌ Error: {MAPPINGS_FILE} not found. Ranker requires this to function.")
        sys.exit(1)

    # 2. Run Performance Gate (Latency & MAE)
    print("📊 Running performance benchmarks...")
    results = benchmark() # Returns ONNX vs PyTorch comparison metrics
    
    mean_latency = results.get("onnx_mean_ms", 999)
    mae = results.get("mae", 1.0)

    # 3. Decision Logic
    passed_latency = mean_latency < LATENCY_THRESHOLD_MS
    passed_accuracy = mae < MAE_THRESHOLD

    if not (passed_latency and passed_accuracy):
        print("❌ PROMOTION REJECTED")
        print(f"Reason: Latency ({mean_latency}ms) or MAE ({mae}) exceeded budget.")
        sys.exit(1)

    # 4. Versioning & Archiving
    os.makedirs(ARCHIVE_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_path = f"{ARCHIVE_DIR}/user_tower_{timestamp}.onnx"
    
    shutil.copy2(SOURCE_MODEL, archive_path)
    print(f"📦 Archived model to: {archive_path}")

    # 5. Promote to Production Path
    # This aligns with the new dynamic path the API will look for
    shutil.copy2(SOURCE_MODEL, PROD_MODEL)
    print(f"🚚 Promoted {SOURCE_MODEL} to {PROD_MODEL}")

    # 6. Update Registry (Generates metadata.json)
    # Includes the explicit production path to be read by the API
    version_tag = f"v{datetime.datetime.now().strftime('%y.%m.%d')}"
    update_model_metadata(
        version=version_tag, 
        metrics=results, 
        model_path=PROD_MODEL
    )

    print(f"🎉 SUCCESS: Model {version_tag} is now live!")
    sys.exit(0)

if __name__ == "__main__":
    publish_production_model()