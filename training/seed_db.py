import pandas as pd
import numpy as np
import asyncio
import os
import sys
from sqlalchemy import insert

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.database import AsyncSessionLocal
from src.database.models import Product, Aisle, Department

async def seed_all():
    path = "./data/raw/"
    df_products = pd.read_csv(f"{path}products.csv")
    df_aisles = pd.read_csv(f"{path}aisles.csv")
    df_depts = pd.read_csv(f"{path}departments.csv")

    # Business Metrics Generation
    np.random.seed(42)
    df_products['price'] = np.random.uniform(5.0, 50.0, size=len(df_products)).round(2)
    df_products['margin'] = (df_products['price'] * np.random.uniform(0.1, 0.3, size=len(df_products))).round(2)
    df_products['stock'] = np.random.randint(50, 501, size=len(df_products))

    async with AsyncSessionLocal() as session:
        try:
            # 1. Seed Metadata (Small files, can be done at once)
            async with session.begin():
                print("📤 Seeding Aisles and Departments...")
                await session.execute(insert(Aisle), df_aisles.to_dict(orient="records"))
                await session.execute(insert(Department), df_depts.to_dict(orient="records"))
            
            # 2. Seed Products in Batches
            product_data = df_products[[
                'product_id', 'product_name', 'aisle_id', 'department_id', 'price', 'margin', 'stock'
            ]].to_dict(orient="records")
            
            batch_size = 5000
            print(f"🚀 Pushing {len(product_data)} products in batches of {batch_size}...")
            
            for i in range(0, len(product_data), batch_size):
                batch = product_data[i : i + batch_size]
                async with session.begin():
                    await session.execute(insert(Product), batch)
                print(f"✅ Indexed {i + len(batch)} / {len(product_data)} products...")

            await session.commit()
            print("🎉 Database fully seeded!")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            await session.rollback()

if __name__ == "__main__":
    asyncio.run(seed_all())