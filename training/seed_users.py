import pandas as pd
import asyncio
import os
import sys
from sqlalchemy import insert

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.database import AsyncSessionLocal
from src.database.models import UserProfile

async def seed_users():
    file_path = "./data/processed/user_profiles.csv"
    if not os.path.exists(file_path):
        print("❌ Error: user_profiles.csv not found.")
        return

    df = pd.read_csv(file_path)
    
    # We only take columns that actually exist in the CSV
    # SQLAlchemy will fill 'last_active' with the default UTC now automatically
    user_data = df[['user_id', 'avg_margin_preference', 'total_purchases']].to_dict(orient="records")

    async with AsyncSessionLocal() as session:
        try:
            batch_size = 5000 # Smaller batches for stability
            print(f"🚀 Pushing {len(user_data)} users to Neon...")
            
            for i in range(0, len(user_data), batch_size):
                batch = user_data[i : i + batch_size]
                async with session.begin():
                    await session.execute(insert(UserProfile), batch)
                print(f"✅ Batch {i//batch_size + 1} complete ({i + len(batch)} users total)")
            
            await session.commit()
            print("🎉 User seeding successful!")
        except Exception as e:
            print(f"❌ Seeding failed: {e}")
            await session.rollback()

if __name__ == "__main__":
    asyncio.run(seed_users())