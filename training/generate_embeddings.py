import torch
import pandas as pd
from sqlalchemy import create_engine, text
from training.model import TwoTowerModel
import os
from src.core.config import settings # Use centralized settings

def generate_and_upload():
    # Use the raw postgresql:// URL from settings
    DATABASE_URL = settings.DATABASE_URL
    
    if not DATABASE_URL:
        print("❌ Error: DATABASE_URL not found in settings.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Generating vectors on: {device}")

    verified_num_users = 206210
    verified_num_products = 49688 
    
    model = TwoTowerModel(num_users=verified_num_users, num_products=verified_num_products).to(device)
    model.load_state_dict(torch.load("models/two_tower_best.pth"))
    model.eval()

    products_df = pd.read_csv("./data/raw/products.csv")
    product_indices = torch.arange(verified_num_products + 1).to(device) 
    
    with torch.no_grad():
        embeddings = model.product_fc(model.product_embedding(product_indices))
        embeddings = embeddings.cpu().numpy()

    # Synchronous engine for bulk updates
    engine = create_engine(settings.SYNC_DATABASE_URL, connect_args={'connect_timeout': 60})
    print(f"📡 Connecting to database to update {len(products_df)} products...")

    batch_size = 2000 
    with engine.begin() as conn:
        for i in range(0, len(products_df), batch_size):
            batch_df = products_df.iloc[i : i + batch_size]
            for offset, (idx, row) in enumerate(batch_df.iterrows()):
                pid = int(row['product_id'])
                vector_index = i + offset
                vector_str = str(embeddings[vector_index].tolist())
                
                conn.execute(
                    text("UPDATE products SET embedding = :vec WHERE product_id = :pid"),
                    {"vec": vector_str, "pid": pid}
                )
            print(f"📤 Progress: {min(i + batch_size, len(products_df))} / {len(products_df)} uploaded")

if __name__ == "__main__":
    generate_and_upload()