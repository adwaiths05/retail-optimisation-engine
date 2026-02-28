import pandas as pd
import xgboost as xgb
import joblib
import torch
import numpy as np
import os
from training.model import TwoTowerModel
from tqdm import tqdm  # Recommended for tracking progress

K_RETRIEVAL = 200
CHECKPOINT_PATH = "models/two_tower_best.pth"

def train_reranker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Reranker on {device}...")

    # 1️⃣ Load checkpoint to get exact layer shapes
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    target_u_size = checkpoint["user_embedding.weight"].shape[0] # 206211
    target_p_size = checkpoint["product_embedding.weight"].shape[0] # 49689

    # ------------------------------------------------------------------
    # FIX: Subtracting 1 because the Model Class likely adds +1 internally
    # We want the final layer to be EXACTLY target_u_size
    # ------------------------------------------------------------------
    model = TwoTowerModel(target_u_size - 1, target_p_size - 1).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"✅ Model loaded. Final layers: {target_u_size} users, {target_p_size} products.")

    # 2️⃣ Load data
    print("📖 Loading datasets...")
    orders = pd.read_csv("./data/raw/orders.csv")
    products = pd.read_csv("./data/raw/products.csv")
    prior = pd.read_csv("./data/raw/order_products__prior.csv").head(2000000) # Sample for speed

    order_to_user = orders.set_index("order_id")["user_id"].to_dict()

    # 3️⃣ Build Mappings
    # We must ensure indices are within [0, target_size - 1]
    user_map = {uid: i for i, uid in enumerate(orders["user_id"].unique()) if i < target_u_size}
    prod_map = {pid: i for i, pid in enumerate(products["product_id"].unique()) if i < target_p_size}

    os.makedirs("models", exist_ok=True)
    joblib.dump({"user_map": user_map, "prod_map": prod_map}, "models/mappings.pkl")

    prior["u_idx"] = prior["order_id"].map(order_to_user).map(user_map)
    prior["p_idx"] = prior["product_id"].map(prod_map)
    prior = prior.dropna(subset=["u_idx", "p_idx"])
    prior["u_idx"] = prior["u_idx"].astype(int)
    prior["p_idx"] = prior["p_idx"].astype(int)

    # 4️⃣ Precompute product embeddings
    valid_indices = torch.arange(target_p_size).to(device)
    with torch.no_grad():
        all_product_vecs = model.product_fc(model.product_embedding(valid_indices))

    # 5️⃣ Generate training samples (Optimized)
    print("🧠 Generating retrieval-based training samples...")
    grouped_purchases = prior.groupby("u_idx")["p_idx"].apply(set).to_dict()
    # Fast User-Item Frequency lookup
    user_item_counts = prior.groupby(["u_idx", "p_idx"]).size().to_dict()

    training_data = []
    sample_users = list(grouped_purchases.keys())[:15000]

    with torch.no_grad():
        for u_idx in tqdm(sample_users):
            u_tensor = torch.tensor([u_idx]).to(device)
            u_vec = model.user_fc(model.user_embedding(u_tensor))
            scores = torch.matmul(u_vec, all_product_vecs.T).squeeze(0)

            topk = torch.topk(scores, min(K_RETRIEVAL, target_p_size))
            candidate_indices = topk.indices.cpu().numpy()
            candidate_scores = topk.values.cpu().numpy()

            purchased_set = grouped_purchases[u_idx]

            for i, p_idx in enumerate(candidate_indices):
                label = 1 if p_idx in purchased_set else 0
                # Get frequency feature
                freq = user_item_counts.get((u_idx, p_idx), 0)
                
                training_data.append({
                    "retrieval_score": candidate_scores[i],
                    "purchase_count": freq,
                    "target": label
                })

    df = pd.DataFrame(training_data)

    # 6️⃣ Train XGBoost
    print("📈 Training XGBoost reranker...")
    X = df[["purchase_count", "retrieval_score"]]
    y = df["target"]

    # Use scale_pos_weight to handle class imbalance (mostly 0s in candidates)
    pos_weight = (len(y) - sum(y)) / sum(y) if sum(y) > 0 else 1

    ranker = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=pos_weight,
        eval_metric="logloss"
    )

    ranker.fit(X, y)
    joblib.dump(ranker, "models/reranker_xgb.pkl")
    print("✅ Reranker and mappings saved successfully!")

if __name__ == "__main__":
    train_reranker()