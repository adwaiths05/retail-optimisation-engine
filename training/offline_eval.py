import json
import os

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

K_RETRIEVAL = 80
K_FINAL = 10


def calculate_ndcg(actual, predicted, k):
    if not actual:
        return 0.0
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    dcg = sum(
        1.0 / np.log2(i + 2)
        if (i < len(predicted) and predicted[i] in actual)
        else 0.0
        for i in range(k)
    )
    return dcg / idcg if idcg > 0 else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _metrics_block(all_actual, all_predicted):
    precisions = []
    recalls = []
    ndcgs = []
    for actual, predicted in zip(all_actual, all_predicted):
        hits = len(set(predicted[:K_FINAL]) & set(actual))
        precisions.append(_safe_div(hits, K_FINAL))
        recalls.append(_safe_div(hits, len(actual)))
        ndcgs.append(calculate_ndcg(actual, predicted[:K_FINAL], K_FINAL))
    return {
        "precision@10": float(np.mean(precisions) if precisions else 0.0),
        "recall@10": float(np.mean(recalls) if recalls else 0.0),
        "ndcg@10": float(np.mean(ndcgs) if ndcgs else 0.0),
    }


def evaluate():
    print("📊 Running staged ranking evaluation...")

    orders = pd.read_csv("./data/raw/orders.csv", usecols=["order_id", "user_id", "order_number"])
    train_orders = pd.read_csv("./data/raw/order_products__train.csv", usecols=["order_id", "product_id"])
    prior = pd.read_csv(
        "./data/raw/order_products__prior.csv",
        usecols=["order_id", "product_id", "reordered"],
    ).head(5000000)
    products = pd.read_csv("./data/raw/products.csv", usecols=["product_id", "aisle_id", "department_id"])

    order_to_user = orders.set_index("order_id")["user_id"].to_dict()
    order_to_number = orders.set_index("order_id")["order_number"].to_dict()

    prior["user_id"] = prior["order_id"].map(order_to_user)
    prior["order_number"] = prior["order_id"].map(order_to_number)
    prior = prior.merge(products, on="product_id", how="left")
    prior = prior.dropna(subset=["user_id", "product_id", "order_number", "aisle_id", "department_id"])

    grouped_user_counts = (
        prior.groupby(["user_id", "product_id"]).size().reset_index(name="cnt")
    )
    grouped_user_counts = grouped_user_counts.sort_values(["user_id", "cnt"], ascending=[True, False])
    user_top_items = grouped_user_counts.groupby("user_id")["product_id"].apply(list).to_dict()

    user_total_interactions = prior.groupby("user_id").size().to_dict()
    user_max_order = prior.groupby("user_id")["order_number"].max().to_dict()
    pair_count = prior.groupby(["user_id", "product_id"]).size().to_dict()
    pair_last_order = prior.groupby(["user_id", "product_id"])["order_number"].max().to_dict()
    product_reorder_ratio = prior.groupby("product_id")["reordered"].mean().to_dict()
    dept_affinity = {
        (u, d): _safe_div(c, user_total_interactions.get(u, 0))
        for (u, d), c in prior.groupby(["user_id", "department_id"]).size().to_dict().items()
    }
    aisle_affinity = {
        (u, a): _safe_div(c, user_total_interactions.get(u, 0))
        for (u, a), c in prior.groupby(["user_id", "aisle_id"]).size().to_dict().items()
    }

    product_lookup = products.set_index("product_id").to_dict("index")
    global_popularity = prior["product_id"].value_counts().index.tolist()

    ml_bundle = joblib.load("models/reranker_xgb.pkl")
    ml_model = ml_bundle["model"]
    feature_cols = ml_bundle["feature_columns"]

    ground_truth = train_orders.groupby("order_id")["product_id"].apply(list).to_dict()
    sample_orders = list(ground_truth.items())[:500]

    retrieval_preds = []
    heuristic_preds = []
    ml_preds = []
    all_actual = []

    for order_id, actual_pids in tqdm(sample_orders, desc="Evaluating"):
        user_id = order_to_user.get(order_id)
        if user_id is None:
            continue

        base_candidates = user_top_items.get(user_id, [])[:K_RETRIEVAL]
        if len(base_candidates) < K_RETRIEVAL:
            fillers = [pid for pid in global_popularity if pid not in base_candidates]
            base_candidates.extend(fillers[: K_RETRIEVAL - len(base_candidates)])

        retrieval_ranked = base_candidates[:K_FINAL]

        heuristic_scored = []
        ml_feature_rows = []
        for idx, pid in enumerate(base_candidates):
            product_meta = product_lookup.get(pid, {"aisle_id": 0, "department_id": 0})
            p_cnt = pair_count.get((user_id, pid), 0)
            p_last_order = pair_last_order.get((user_id, pid), 0)
            u_total = user_total_interactions.get(user_id, 0)
            u_max_order = user_max_order.get(user_id, 1)
            recency = 1.0 - _safe_div(u_max_order - p_last_order, max(u_max_order, 1))
            cat_aff = (
                dept_affinity.get((user_id, product_meta["department_id"]), 0.0)
                + aisle_affinity.get((user_id, product_meta["aisle_id"]), 0.0)
            ) / 2.0

            retrieval_score = 1.0 - _safe_div(idx, K_RETRIEVAL)
            margin_sim = 0.2
            inventory_sim = 0.5

            heur_score = (
                0.9 * retrieval_score
                + 0.05 * margin_sim
                + 0.05 * inventory_sim
            )
            heuristic_scored.append((pid, heur_score))

            ml_feature_rows.append(
                {
                    "retrieval_score": retrieval_score,
                    "embedding_similarity": retrieval_score,
                    "user_product_frequency": float(p_cnt),
                    "user_product_reorder_ratio": float(product_reorder_ratio.get(pid, 0.0)),
                    "user_product_recency": float(max(0.0, min(recency, 1.0))),
                    "category_affinity": float(cat_aff),
                    "user_item_share": _safe_div(p_cnt, u_total),
                    "price": 4.0,
                    "margin": 1.5,
                    "stock": 120.0,
                    "pid": pid,
                }
            )

        heuristic_ranked = [pid for pid, _ in sorted(heuristic_scored, key=lambda x: x[1], reverse=True)[:K_FINAL]]

        ml_frame = pd.DataFrame(ml_feature_rows)
        ml_scores = ml_model.predict(ml_frame[feature_cols])
        ml_frame["ml_score"] = ml_scores
        ml_ranked = ml_frame.sort_values("ml_score", ascending=False)["pid"].tolist()[:K_FINAL]

        retrieval_preds.append(retrieval_ranked)
        heuristic_preds.append(heuristic_ranked)
        ml_preds.append(ml_ranked)
        all_actual.append(actual_pids)

    retrieval_metrics = _metrics_block(all_actual, retrieval_preds)
    heuristic_metrics = _metrics_block(all_actual, heuristic_preds)
    ml_metrics = _metrics_block(all_actual, ml_preds)

    summary = {
        "retrieval_only": retrieval_metrics,
        "retrieval_plus_heuristic_rerank": heuristic_metrics,
        "retrieval_plus_ml_rerank": ml_metrics,
    }

    print("\n" + "=" * 52)
    print("📌 RANKING PIPELINE COMPARISON")
    print("=" * 52)
    for name, metrics in summary.items():
        print(name)
        print(f"  Precision@10: {metrics['precision@10']:.4f}")
        print(f"  Recall@10:    {metrics['recall@10']:.4f}")
        print(f"  NDCG@10:      {metrics['ndcg@10']:.4f}")
        print("-" * 52)

    os.makedirs("mlops/reports", exist_ok=True)
    with open("mlops/reports/offline_ranking_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("✅ Saved staged metrics to mlops/reports/offline_ranking_metrics.json")

if __name__ == "__main__":
    evaluate()