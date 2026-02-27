import pandas as pd
import numpy as np
import os

def generate_user_features():
    raw_path = "./data/raw/"
    processed_path = "./data/processed/"
    os.makedirs(processed_path, exist_ok=True)

    print("📖 Loading orders and products for cross-referencing...")
    # We need products to know the 'margin' of what they bought
    products = pd.read_csv(f"{raw_path}products.csv")
    
    # We regenerate the same synthetic margins used in seeding so data matches
    np.random.seed(42)
    products['price'] = np.random.uniform(5.0, 50.0, size=len(products)).round(2)
    products['margin'] = (products['price'] * np.random.uniform(0.1, 0.3, size=len(products))).round(2)
    
    # Just keep what we need to save memory
    margin_map = products.set_index('product_id')['margin'].to_dict()

    print("⚡ Processing 32M+ rows of order_products (this may take a minute)...")
    # We use chunks if your RAM is under 16GB. 
    # If you have 16GB+, you can read it directly.
    chunk_size = 1_000_000
    user_margins = {}
    user_counts = {}

    # Merge orders.csv with order_products__prior.csv to link User_ID to Product_ID
    orders = pd.read_csv(f"{raw_path}orders.csv")[[ 'order_id', 'user_id']]
    order_to_user = orders.set_index('order_id')['user_id'].to_dict()

    reader = pd.read_csv(f"{raw_path}order_products__prior.csv", chunksize=chunk_size)
    
    for i, chunk in enumerate(reader):
        # Map product to its margin
        chunk['margin'] = chunk['product_id'].map(margin_map)
        # Map order to its user
        chunk['user_id'] = chunk['order_id'].map(order_to_user)
        
        # Aggregate in memory
        agg = chunk.groupby('user_id')['margin'].agg(['sum', 'count'])
        
        for user_id, row in agg.iterrows():
            user_margins[user_id] = user_margins.get(user_id, 0) + row['sum']
            user_counts[user_id] = user_counts.get(user_id, 0) + row['count']
        
        if i % 5 == 0:
            print(f"✅ Processed {i} million rows...")

    print("📊 Finalizing User Profiles...")
    user_features = pd.DataFrame({
        'user_id': list(user_margins.keys()),
        'avg_margin_preference': [user_margins[u] / user_counts[u] for u in user_margins],
        'total_purchases': list(user_counts.values())
    })

    # Save to processed folder
    save_file = f"{processed_path}user_profiles.csv"
    user_features.to_csv(save_file, index=False)
    print(f"✨ Success! Saved {len(user_features)} user profiles to {save_file}")

if __name__ == "__main__":
    generate_user_features()