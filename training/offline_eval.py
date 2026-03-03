import pandas as pd
import numpy as np
import os
from tqdm import tqdm

K_FINAL = 10

def calculate_ndcg(actual, predicted, k):
    if not actual: return 0.0
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    dcg = sum(1.0 / np.log2(i + 2) if (i < len(predicted) and predicted[i] in actual) else 0.0 for i in range(k))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate():
    print("📊 Running Heuristic-Boosted Evaluation...")

    # 1. Load Data
    orders = pd.read_csv("./data/raw/orders.csv")
    train_orders = pd.read_csv("./data/raw/order_products__train.csv")
    prior = pd.read_csv("./data/raw/order_products__prior.csv").head(5000000) # Use more data for stats
    
    order_to_user = orders.set_index('order_id')['user_id'].to_dict()
    
    # 2. Build High-Confidence Features
    print("🧹 Building User-Item History and Global Popularity...")
    prior['user_id'] = prior['order_id'].map(order_to_user)
    
    # Feature 1: Top items for EACH user (Personalized)
    user_top_items = prior.groupby('user_id')['product_id'].apply(lambda x: x.value_counts().index[:50].tolist()).to_dict()
    
    # Feature 2: Top items OVERALL (Non-Personalized/Global)
    global_top_items = prior['product_id'].value_counts().index[:100].tolist()

    ground_truth = train_orders.groupby('order_id')['product_id'].apply(list).to_dict()
    sample_orders = list(ground_truth.items())[:500]

    precisions, recalls, ndcgs = [], [], []

    print(f"🧪 Scoring {len(sample_orders)} users using Hybrid Heuristics...")
    for order_id, actual_pids in tqdm(sample_orders):
        user_id = order_to_user.get(order_id)
        
        # LOGIC:
        # 1. Take the items this user buys most frequently (The "Loyalty" set)
        # 2. Fill the remaining slots with Global Top items (The "Trending" set)
        
        preds = user_top_items.get(user_id, [])
        if len(preds) < K_FINAL:
            # Fill with global items not already in the list
            fillers = [i for i in global_top_items if i not in preds]
            preds.extend(fillers)
        
        preds = preds[:K_FINAL]

        # Metrics
        hits = len(set(preds) & set(actual_pids))
        precisions.append(hits / K_FINAL)
        recalls.append(hits / len(actual_pids) if actual_pids else 0)
        ndcgs.append(calculate_ndcg(actual_pids, preds, K_FINAL))

    print("\n" + "="*35)
    print("🚀 HEURISTIC PIPELINE RESULTS")
    print("="*35)
    print(f"Precision@10: {np.mean(precisions):.4f}")
    print(f"Recall@10:    {np.mean(recalls):.4f}")
    print(f"NDCG@10:      {np.mean(ndcgs):.4f}")
    print("="*35)

if __name__ == "__main__":
    evaluate()