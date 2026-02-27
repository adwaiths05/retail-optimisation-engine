import torch
import pandas as pd
from sqlalchemy import create_engine, text
from training.model import TwoTowerModel
import os

# 1. Load the URL from your environment
from dotenv import load_dotenv  # <--- Add this

# 1. Manually trigger the .env file loading
load_dotenv() 

DATABASE_URL = os.getenv("DATABASE_URL")

def generate_and_upload():
    # Safety check for the environment variable
    if not DATABASE_URL:
        print("❌ Error: DATABASE_URL environment variable not found.")
        print("Please run: $env:DATABASE_URL='your_neon_url' before executing.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Generating vectors on: {torch.cuda.get_device_name(0)}")

    # 2. Match your Verification Results exactly
    verified_num_users = 206210
    verified_num_products = 49688
    
    # Initialize model with verified dimensions
    model = TwoTowerModel(num_users=verified_num_users, num_products=verified_num_products).to(device)
    
    # Load weights
    model.load_state_dict(torch.load("models/two_tower_best.pth"))
    model.eval()
    print("💾 Model weights loaded successfully.")

    # 3. Load product list to map IDs correctly
    products_df = pd.read_csv("./data/raw/products.csv")
    
    # Create the same range of indices used during training
    product_indices = torch.arange(len(products_df)).to(device)
    
    print("🧠 Extracting Product Tower embeddings...")
    with torch.no_grad():
        embeddings = model.product_fc(model.product_embedding(product_indices))
        embeddings = embeddings.cpu().numpy()

    # 4. Upload to Neon
    engine = create_engine(DATABASE_URL)
    print(f"📡 Connecting to Neon to update {len(products_df)} products...")

    batch_size = 200 
    with engine.begin() as conn:
        for i in range(0, len(products_df), batch_size):
            batch_df = products_df.iloc[i : i + batch_size]
            for idx, row in batch_df.iterrows():
                pid = int(row['product_id'])
                vector_str = str(embeddings[idx].tolist())
                
                conn.execute(
                    text("UPDATE products SET embedding = :vec WHERE product_id = :pid"),
                    {"vec": vector_str, "pid": pid}
                )
            print(f"📤 Progress: {min(i + batch_size, len(products_df))} / {len(products_df)} uploaded")

    print("\n✨ SUCCESS! Your Neon database is now AI-indexed.")

if __name__ == "__main__":
    generate_and_upload()