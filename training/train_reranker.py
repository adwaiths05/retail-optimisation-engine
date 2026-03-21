import os
from typing import Dict, List

import joblib
import numpy as np
import onnxmltools
import pandas as pd
import torch
import xgboost as xgb
from onnxmltools.convert.common.data_types import FloatTensorType
from tqdm import tqdm

from training.model import TwoTowerModel

K_RETRIEVAL = 200
MAX_USERS = 15000
CHECKPOINT_PATH = "models/two_tower_best.pth"


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _build_feature_store(prior_df: pd.DataFrame) -> Dict[str, Dict]:
    pair_stats = prior_df.groupby(["user_id", "product_id"]).agg(
        interaction_count=("order_id", "size"),
        last_order_number=("order_number", "max"),
    )
    user_total_interactions = prior_df.groupby("user_id").size().to_dict()
    user_max_order = prior_df.groupby("user_id")["order_number"].max().to_dict()
    product_reorder_ratio = prior_df.groupby("product_id")["reordered"].mean().to_dict()

    user_dept_affinity = prior_df.groupby(["user_id", "department_id"]).size().to_dict()
    user_aisle_affinity = prior_df.groupby(["user_id", "aisle_id"]).size().to_dict()

    pair_count = {
        f"{int(u)}:{int(p)}": int(c)
        for (u, p), c in pair_stats["interaction_count"].to_dict().items()
    }
    pair_last_order = {
        f"{int(u)}:{int(p)}": int(o)
        for (u, p), o in pair_stats["last_order_number"].to_dict().items()
    }
    dept_aff = {
        f"{int(u)}:{int(d)}": _safe_div(c, user_total_interactions.get(int(u), 0))
        for (u, d), c in user_dept_affinity.items()
    }
    aisle_aff = {
        f"{int(u)}:{int(a)}": _safe_div(c, user_total_interactions.get(int(u), 0))
        for (u, a), c in user_aisle_affinity.items()
    }

    return {
        "pair_count": pair_count,
        "pair_last_order": pair_last_order,
        "user_total_interactions": {int(k): int(v) for k, v in user_total_interactions.items()},
        "user_max_order": {int(k): int(v) for k, v in user_max_order.items()},
        "product_reorder_ratio": {int(k): float(v) for k, v in product_reorder_ratio.items()},
        "dept_affinity": dept_aff,
        "aisle_affinity": aisle_aff,
    }


def _build_training_rows(
    model: TwoTowerModel,
    device: torch.device,
    target_p_size: int,
    grouped_purchases: Dict[int, set],
    sample_users: List[int],
    idx_to_user: Dict[int, int],
    idx_to_prod: Dict[int, int],
    product_meta: Dict[int, Dict[str, float]],
    feature_store: Dict[str, Dict],
) -> pd.DataFrame:
    valid_indices = torch.arange(target_p_size).to(device)
    with torch.no_grad():
        all_product_vecs = model.product_fc(model.product_embedding(valid_indices))

    rows = []
    with torch.no_grad():
        for u_idx in tqdm(sample_users):
            u_tensor = torch.tensor([u_idx]).to(device)
            u_vec = model.user_fc(model.user_embedding(u_tensor))
            scores = torch.matmul(u_vec, all_product_vecs.T).squeeze(0)
            topk = torch.topk(scores, min(K_RETRIEVAL, target_p_size))

            candidate_indices = topk.indices.cpu().numpy()
            candidate_scores = topk.values.cpu().numpy()
            purchased_set = grouped_purchases[u_idx]

            raw_user_id = int(idx_to_user[u_idx])
            user_total = feature_store["user_total_interactions"].get(raw_user_id, 0)
            user_max_order = feature_store["user_max_order"].get(raw_user_id, 1)

            for rank_i, p_idx in enumerate(candidate_indices):
                raw_product_id = int(idx_to_prod[int(p_idx)])
                pair_key = f"{raw_user_id}:{raw_product_id}"

                pair_count = feature_store["pair_count"].get(pair_key, 0)
                pair_last_order = feature_store["pair_last_order"].get(pair_key, 0)

                recency = 1.0 - _safe_div(user_max_order - pair_last_order, max(user_max_order, 1))
                reorder_ratio = feature_store["product_reorder_ratio"].get(raw_product_id, 0.0)
                dept_id = int(product_meta[raw_product_id]["department_id"])
                aisle_id = int(product_meta[raw_product_id]["aisle_id"])
                category_affinity = (
                    feature_store["dept_affinity"].get(f"{raw_user_id}:{dept_id}", 0.0)
                    + feature_store["aisle_affinity"].get(f"{raw_user_id}:{aisle_id}", 0.0)
                ) / 2.0
                embedding_similarity = float(candidate_scores[rank_i])

                rows.append(
                    {
                        "group_user": raw_user_id,
                        "target": 1 if int(p_idx) in purchased_set else 0,
                        "retrieval_score": embedding_similarity,
                        "embedding_similarity": embedding_similarity,
                        "user_product_frequency": float(pair_count),
                        "user_product_reorder_ratio": float(reorder_ratio),
                        "user_product_recency": float(max(0.0, min(recency, 1.0))),
                        "category_affinity": float(category_affinity),
                        "user_item_share": _safe_div(pair_count, user_total),
                        "price": float(product_meta[raw_product_id]["price"]),
                        "margin": float(product_meta[raw_product_id]["margin"]),
                        "stock": float(product_meta[raw_product_id]["stock"]),
                    }
                )

    return pd.DataFrame(rows)


def train_reranker():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training Ranking Reranker on {device}...")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    target_u_size = checkpoint["user_embedding.weight"].shape[0]
    target_p_size = checkpoint["product_embedding.weight"].shape[0]

    model = TwoTowerModel(target_u_size - 1, target_p_size - 1).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    print("📖 Loading datasets...")
    orders = pd.read_csv("./data/raw/orders.csv", usecols=["order_id", "user_id", "order_number"])
    products = pd.read_csv(
        "./data/raw/products.csv",
        usecols=["product_id", "aisle_id", "department_id"],
    )
    prior = pd.read_csv(
        "./data/raw/order_products__prior.csv",
        usecols=["order_id", "product_id", "reordered"],
    ).head(3000000)

    order_user = orders.set_index("order_id")["user_id"]
    order_number = orders.set_index("order_id")["order_number"]

    prior["user_id"] = prior["order_id"].map(order_user)
    prior["order_number"] = prior["order_id"].map(order_number)
    prior = prior.merge(products, on="product_id", how="left")
    prior = prior.dropna(subset=["user_id", "product_id", "order_number", "aisle_id", "department_id"])

    user_ids = orders["user_id"].dropna().unique().tolist()
    product_ids = products["product_id"].dropna().unique().tolist()

    user_map = {int(uid): i for i, uid in enumerate(user_ids) if i < target_u_size}
    prod_map = {int(pid): i for i, pid in enumerate(product_ids) if i < target_p_size}
    idx_to_user = {idx: uid for uid, idx in user_map.items()}
    idx_to_prod = {idx: pid for pid, idx in prod_map.items()}

    os.makedirs("models", exist_ok=True)
    joblib.dump({"user_map": user_map, "prod_map": prod_map}, "models/mappings.pkl")

    prior["u_idx"] = prior["user_id"].map(user_map)
    prior["p_idx"] = prior["product_id"].map(prod_map)
    prior = prior.dropna(subset=["u_idx", "p_idx"]).copy()
    prior["u_idx"] = prior["u_idx"].astype(int)
    prior["p_idx"] = prior["p_idx"].astype(int)

    feature_store = _build_feature_store(prior)
    joblib.dump(feature_store, "models/reranker_features.pkl")

    grouped_purchases = prior.groupby("u_idx")["p_idx"].apply(set).to_dict()
    sample_users = list(grouped_purchases.keys())[:MAX_USERS]

    product_meta = {}
    grouped_meta = prior.groupby("product_id").agg(
        aisle_id=("aisle_id", "first"),
        department_id=("department_id", "first"),
    )
    for pid, row in grouped_meta.iterrows():
        product_meta[int(pid)] = {
            "aisle_id": int(row["aisle_id"]),
            "department_id": int(row["department_id"]),
            "price": 4.0,
            "margin": 1.5,
            "stock": 120.0,
        }

    print("🧠 Generating ranking training samples...")
    df = _build_training_rows(
        model=model,
        device=device,
        target_p_size=target_p_size,
        grouped_purchases=grouped_purchases,
        sample_users=sample_users,
        idx_to_user=idx_to_user,
        idx_to_prod=idx_to_prod,
        product_meta=product_meta,
        feature_store=feature_store,
    )

    feature_cols = [
        "retrieval_score",
        "embedding_similarity",
        "user_product_frequency",
        "user_product_reorder_ratio",
        "user_product_recency",
        "category_affinity",
        "user_item_share",
        "price",
        "margin",
        "stock",
    ]

    print("📈 Training XGBoost ranker (objective=rank:ndcg)...")
    df = df.sort_values("group_user").reset_index(drop=True)
    group_sizes = df.groupby("group_user").size().tolist()

    X = df[feature_cols]
    y = df["target"]

    ranker = xgb.XGBRanker(
        objective="rank:ndcg",
        n_estimators=250,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        random_state=42,
        tree_method="hist",
    )
    ranker.fit(X, y, group=group_sizes, verbose=False)

    joblib.dump(
        {
            "model": ranker,
            "feature_columns": feature_cols,
        },
        "models/reranker_xgb.pkl",
    )

    try:
        print("🚀 Exporting ranker to ONNX...")
        ranker.get_booster().feature_names = None
        initial_type = [("input", FloatTensorType([None, len(feature_cols)]))]
        onnx_model = onnxmltools.convert_xgboost(ranker, initial_types=initial_type, target_opset=12)
        onnxmltools.utils.save_model(onnx_model, "models/reranker.onnx")
    except Exception as exc:
        print(f"⚠️ ONNX export skipped: {exc}")

    print("✅ Ranking reranker artifacts saved.")

if __name__ == "__main__":
    train_reranker()