import joblib
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import os

# 1. Load the existing XGBoost pickle model
model_path = "models/reranker_xgb.pkl"
if not os.path.exists(model_path):
    print(f"❌ Error: {model_path} not found.")
    exit(1)

xgboost_model = joblib.load(model_path)

# --- FIX: Remove Feature Names ---
# ONNX converter struggles with string feature names like 'purchase_count'.
# We tell the underlying booster to forget the names so it uses f0, f1.
xgboost_model.get_booster().feature_names = None

# 2. Define the input schema
# [None, 2] means dynamic number of rows, and exactly 2 features (f0, f1)
initial_type = [('input', FloatTensorType([None, 2]))]

# 3. Convert the model to ONNX
print("🚀 Converting XGBoost model to ONNX...")
onnx_model = onnxmltools.convert_xgboost(
    xgboost_model, 
    initial_types=initial_type,
    target_opset=12
)

# 4. Save the ONNX model
output_path = "models/reranker.onnx"
onnxmltools.utils.save_model(onnx_model, output_path)

print(f"✅ Success! Reranker converted and saved to {output_path}")