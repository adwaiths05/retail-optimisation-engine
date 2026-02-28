import torch
import onnx
from training.model import TwoTowerModel
import os

def export():
    # 1. Initialize model with your verified dimensions
    num_users = 206210 + 1 
    num_products = 49688 + 1
    model = TwoTowerModel(num_users, num_products)
    
    # 2. Load trained weights
    model.load_state_dict(torch.load("models/two_tower_best.pth", map_location="cpu"))
    model.eval()

    # 3. Create a dummy input (a single User ID)
    dummy_user_id = torch.tensor([1], dtype=torch.long)

    # 4. Define a wrapper to export ONLY the User Tower logic
    class UserTowerWrapper(torch.nn.Module):
        def __init__(self, user_tower_model):
            super().__init__()
            self.user_embedding = user_tower_model.user_embedding
            self.user_fc = user_tower_model.user_fc
        
        def forward(self, user_id):
            return self.user_fc(self.user_embedding(user_id))

    user_tower = UserTowerWrapper(model)

    # 5. Export to ONNX
    os.makedirs("models", exist_ok=True)
    onnx_path = "models/user_tower.onnx"
    
    torch.onnx.export(
        user_tower,
        dummy_user_id,
        onnx_path,
        input_names=['user_id'],
        output_names=['user_embedding'],
        dynamic_axes={'user_id': {0: 'batch_size'}},
        opset_version=12
    )
    print(f"✅ User Tower exported to {onnx_path}")

if __name__ == "__main__":
    export()