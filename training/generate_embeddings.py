import torch
import pandas as pd
from sqlalchemy import create_engine, text
from training.model import TwoTowerModel
import os
from dotenv import load_dotenv 

load_dotenv() 

DATABASE_URL = os.getenv("DATABASE_URL")

def generate_and_upload():
    if not DATABASE_URL:
        print("❌ Error: DATABASE_URL environment variable not found.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Generating vectors on: {torch.cuda.get_device_name(0)}")

    # Match your Verification Results exactly
    verified_num_users = 206210
    verified_num_products = 49688 # This is the count from your CSV
    
    # Initialize model - model size is count + 1 (49689)
    model = TwoTowerModel(num_users=verified_num_users, num_products=verified_num_products).to(device)
    
    model.load_state_dict(torch.load("models/two_tower_best.pth"))
    model.eval()
    print("💾 Model weights loaded successfully.")

    products_df = pd.read_csv("./data/raw/products.csv")
    
    # Generate all 49,689 embeddings to satisfy the model
    product_indices = torch.arange(verified_num_products + 1).to(device) 
    
    print("🧠 Extracting Product Tower embeddings...")
    with torch.no_grad():
        # Get the output from the Product Tower
        embeddings = model.product_fc(model.product_embedding(product_indices))
        embeddings = embeddings.cpu().numpy()

    engine = create_engine(DATABASE_URL, connect_args={'connect_timeout': 60})
    print(f"📡 Connecting to Neon to update {len(products_df)} products...")

    batch_size = 2000 # Reduced for better stability on Neon
    
    with engine.begin() as conn:
        for i in range(0, len(products_df), batch_size):
            batch_df = products_df.iloc[i : i + batch_size]
            
            for offset, (idx, row) in enumerate(batch_df.iterrows()):
                pid = int(row['product_id'])
                
                # FIX: Use (i + offset) to get the correct vector index
                # This ensures vector 0 goes to row 0, vector 1 to row 1, etc.
                vector_index = i + offset
                vector_str = str(embeddings[vector_index].tolist())
                
                conn.execute(
                    text("UPDATE products SET embedding = :vec WHERE product_id = :pid"),
                    {"vec": vector_str, "pid": pid}
                )
            
            print(f"📤 Progress: {min(i + batch_size, len(products_df))} / {len(products_df)} uploaded")

    print("\n✨ SUCCESS! Your Neon database is now AI-indexed.")

if __name__ == "__main__":
    generate_and_upload()