import pandas as pd
import torch
from torch.utils.data import Dataset
import os

class RetailDataset(Dataset):
    def __init__(self, interactions_path, products_path):
        print(f"📡 Loading interaction data...")
        
        # Load interactions
        self.interactions = pd.read_csv(interactions_path, usecols=['order_id', 'product_id'])
        
        # Load orders to map order_id -> user_id
        orders_path = os.path.join(os.path.dirname(interactions_path), "orders.csv")
        orders = pd.read_csv(orders_path, usecols=['order_id', 'user_id'])
        
        # Merge to get user_id
        self.interactions = self.interactions.merge(orders, on='order_id')
        
        # Load products for mapping
        products = pd.read_csv(products_path, usecols=['product_id'])
        self.product_list = products['product_id'].unique()
        self.product_map = {pid: i for i, pid in enumerate(self.product_list)}
        self.num_products = len(self.product_list)
        
        print(f"✅ Dataset ready: {len(self.interactions)} rows.")

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        row = self.interactions.iloc[idx]
        user_id = int(row['user_id'])
        product_idx = self.product_map.get(int(row['product_id']), 0)
        
        return {
            'user_id': torch.tensor(user_id, dtype=torch.long),
            'product_idx': torch.tensor(product_idx, dtype=torch.long)
        }