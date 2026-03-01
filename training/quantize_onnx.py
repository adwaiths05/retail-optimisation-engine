import os
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
def quantize_model():
    input_model_path = "models/user_tower.onnx"
    output_model_path = "models/user_tower_quantized.onnx"
    
    # Ensure the models directory exists
    if not os.path.exists("models"):
        print("📁 Creating 'models' directory...")
        os.makedirs("models")

    if not os.path.exists(input_model_path):
        print(f"❌ Error: {input_model_path} not found.")
        return

    print(f"⚡ Quantizing {input_model_path}...")
    try:
        quantize_dynamic(
            model_input=input_model_path,
            model_output=output_model_path,
            weight_type=QuantType.QUInt8
        )
        print(f"✅ Success! Created: {output_model_path}")
    except Exception as e:
        print(f"❌ Quantization failed: {e}")

if __name__ == "__main__":
    quantize_model()