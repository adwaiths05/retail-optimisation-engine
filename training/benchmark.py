import torch
import onnxruntime as ort
import numpy as np
import time
import os
from training.model import TwoTowerModel #

def benchmark():
    # --- 1. SETUP & CONFIG ---
    model_path_pth = "models/two_tower_best.pth" #
    model_path_onnx = "models/user_tower_quantized.onnx"   #
    num_users = 206210 
    num_products = 49688 
    iterations = 1000  # Number of runs for a stable average
    
    # --- 2. MODEL SIZE MEASUREMENT ---
    size_pth = os.path.getsize(model_path_pth) / (1024 * 1024)
    size_onnx = os.path.getsize(model_path_onnx) / (1024 * 1024)
    print(f"📊 MODEL SIZE: PyTorch: {size_pth:.2f}MB | ONNX: {size_onnx:.2f}MB")

    # --- 3. PREPARE MODELS ---
    # PyTorch User Tower
    full_model = TwoTowerModel(num_users, num_products) #
    full_model.load_state_dict(torch.load(model_path_pth, map_location="cpu")) #
    full_model.eval()
    
    class UserTowerOnly(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.emb = m.user_embedding
            self.fc = m.user_fc
        def forward(self, x): return self.fc(self.emb(x))

    torch_user_tower = UserTowerOnly(full_model)
    
    # ONNX Session
    ort_session = ort.InferenceSession(model_path_onnx) #
    input_name = ort_session.get_inputs()[0].name

    # --- 4. LATENCY & NUMERICAL PARITY ---
    dummy_user_id = 1234
    input_tensor = torch.tensor([dummy_user_id], dtype=torch.long)
    input_numpy = np.array([dummy_user_id], dtype=np.int64)

    # Warm-up (Standard practice for stable benchmarks)
    for _ in range(10):
        _ = torch_user_tower(input_tensor)
        _ = ort_session.run(None, {input_name: input_numpy})

    # Benchmark PyTorch
    pt_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        pt_out = torch_user_tower(input_tensor)
        pt_times.append((time.perf_counter() - start) * 1000)

    # Benchmark ONNX
    onx_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        onx_out = ort_session.run(None, {input_name: input_numpy})
        onx_times.append((time.perf_counter() - start) * 1000)

    # Numerical Parity (MAE)
    mae = np.abs(pt_out.detach().numpy() - onx_out[0]).mean()

    # --- 5. RESULTS ---
    print(f"\n⏱️ LATENCY (ms):")
    print(f"PyTorch -> Mean: {np.mean(pt_times):.4f} | P99: {np.percentile(pt_times, 99):.4f}")
    print(f"ONNX    -> Mean: {np.mean(onx_times):.4f} | P99: {np.percentile(onx_times, 99):.4f}")
    print(f"🚀 Speedup: {np.mean(pt_times) / np.mean(onx_times):.2f}x faster")
    
    print(f"\n✅ NUMERICAL PARITY:")
    print(f"Mean Absolute Error: {mae:.2e} (Should be near zero)")

if __name__ == "__main__":
    benchmark()