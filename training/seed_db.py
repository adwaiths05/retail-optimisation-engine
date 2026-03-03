import pandas as pd
import numpy as np
import os
import sys
from sqlalchemy import create_engine, insert

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.config import settings
from src.database.models import Product, Aisle, Department

def seed_all():
    # Use the synchronous DATABASE_URL
    engine = create_engine(settings.DATABASE_URL)
    
    path = "./data/raw/"
    df_products = pd.read_csv(f"{path}products.csv")
    df_aisles = pd.read_csv(f"{path}aisles.csv")
    df_depts = pd.read_csv(f"{path}departments.csv")

    # Business Metrics Generation
    np.random.seed(42)
    df_products['price'] = np.random.uniform(5.0, 50.0, size=len(df_products)).round(2)
    df_products['margin'] = (df_products['price'] * np.random.uniform(0.1, 0.3, size=len(df_products))).round(2)
    df_products['stock'] = np.random.randint(50, 501, size=len(df_products))

    with engine.begin() as conn:
        try:
            print("📤 Seeding Aisles and Departments...")
            conn.execute(insert(Aisle), df_aisles.to_dict(orient="records"))
            conn.execute(insert(Department), df_depts.to_dict(orient="records"))
            
            product_data = df_products[[
                'product_id', 'product_name', 'aisle_id', 'department_id', 'price', 'margin', 'stock'
            ]].to_dict(orient="records")
            
            batch_size = 5000
            print(f"🚀 Pushing {len(product_data)} products in batches of {batch_size}...")
            
            for i in range(0, len(product_data), batch_size):
                batch = product_data[i : i + batch_size]
                conn.execute(insert(Product), batch)
                print(f"✅ Indexed {i + len(batch)} / {len(product_data)} products...")

            print("🎉 Database fully seeded!")
            
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    seed_all()