import torch
import pandas as pd
import os

def verify():
    model_path = 'models/two_tower_best.pth'
    products_path = './data/raw/products.csv'
    orders_path = './data/raw/orders.csv'

    if not os.path.exists(model_path):
        print(f"❌ Error: {model_path} not found. Did you save the model?")
        return

    # Load data
    products = pd.read_csv(products_path)
    orders = pd.read_csv(orders_path)
    
    # Load model weights (CPU is fine for just checking shapes)
    state_dict = torch.load(model_path, map_location='cpu')

    print("\n" + "="*35)
    print("📊 VERIFICATION RESULTS")
    print("="*35)
    
    csv_products = len(products)
    csv_max_user = orders.user_id.max()
    
    model_products = state_dict['product_embedding.weight'].shape[0]
    model_users = state_dict['user_embedding.weight'].shape[0]

    print(f"CSV Product Count:    {csv_products}")
    print(f"Model Product Size:   {model_products}")
    print("-" * 35)
    print(f"CSV Max User ID:      {csv_max_user}")
    print(f"Model User Size:      {model_users}")
    print("=" * 35)

    if csv_products == model_products:
        print("✅ Product dimensions match!")
    else:
        print("⚠️ Product dimensions MISMATCH!")

    if model_users > csv_max_user:
        print("✅ User dimensions are safe!")
    else:
        print("⚠️ User dimensions might be too small!")

if __name__ == "__main__":
    verify()