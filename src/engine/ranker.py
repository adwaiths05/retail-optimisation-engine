from typing import Any, Dict, List

import joblib
import numpy as np
import onnxruntime as ort


class Ranker:
    def __init__(self, model_path: str, mappings_path: str):
        self.session = None
        self.input_name = None
        self.pkl_model = None
        self.feature_columns: List[str] = []

        try:
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
        except Exception as exc:
            print(f"⚠️ ONNX ranker unavailable: {exc}")

        try:
            model_bundle = joblib.load("models/reranker_xgb.pkl")
            self.pkl_model = model_bundle.get("model")
            self.feature_columns = model_bundle.get("feature_columns", [])
        except Exception as exc:
            print(f"⚠️ PKL ranker unavailable: {exc}")

        try:
            self.feature_store = joblib.load("models/reranker_features.pkl")
        except Exception as exc:
            print(f"⚠️ Feature store unavailable: {exc}")
            self.feature_store = {}

        try:
            self.mappings = joblib.load(mappings_path)
        except Exception:
            self.mappings = {"prod_map": {}, "user_map": {}}

    @staticmethod
    def _safe_div(numerator: float, denominator: float) -> float:
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    def _row_features(self, user_id: int, item: Any) -> Dict[str, float]:
        pair_key = f"{int(user_id)}:{int(item.product_id)}"

        pair_count = self.feature_store.get("pair_count", {}).get(pair_key, 0)
        pair_last_order = self.feature_store.get("pair_last_order", {}).get(pair_key, 0)
        user_total = self.feature_store.get("user_total_interactions", {}).get(int(user_id), 0)
        user_max_order = self.feature_store.get("user_max_order", {}).get(int(user_id), 1)
        reorder_ratio = self.feature_store.get("product_reorder_ratio", {}).get(int(item.product_id), 0.0)

        dept_aff = self.feature_store.get("dept_affinity", {}).get(
            f"{int(user_id)}:{int(getattr(item, 'department_id', 0))}", 0.0
        )
        aisle_aff = self.feature_store.get("aisle_affinity", {}).get(
            f"{int(user_id)}:{int(getattr(item, 'aisle_id', 0))}", 0.0
        )
        recency = 1.0 - self._safe_div(user_max_order - pair_last_order, max(user_max_order, 1))

        return {
            "retrieval_score": float(1.0 - float(item.distance)),
            "embedding_similarity": float(1.0 - float(item.distance)),
            "user_product_frequency": float(pair_count),
            "user_product_reorder_ratio": float(reorder_ratio),
            "user_product_recency": float(max(0.0, min(recency, 1.0))),
            "category_affinity": float((dept_aff + aisle_aff) / 2.0),
            "user_item_share": float(self._safe_div(pair_count, user_total)),
            "price": float(getattr(item, "price", 0.0)),
            "margin": float(getattr(item, "margin", 0.0)),
            "stock": float(getattr(item, "stock", 0.0)),
        }

    def _score_with_ml(self, feature_matrix: np.ndarray) -> np.ndarray:
        if self.session and self.input_name:
            outputs = self.session.run(None, {self.input_name: feature_matrix})
            onnx_scores = outputs[-1]
            return np.array(onnx_scores, dtype=np.float32).reshape(-1)

        if self.pkl_model is not None:
            return np.array(self.pkl_model.predict(feature_matrix), dtype=np.float32).reshape(-1)

        return np.zeros(feature_matrix.shape[0], dtype=np.float32)

    def rank(self, candidates, user_id: int, weights: dict = None):
        if not candidates:
            return []

        if not weights:
            weights = {"w_relevance": 0.92, "w_margin": 0.05, "w_inventory": 0.03}

        default_cols = [
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
        feature_cols = self.feature_columns or default_cols

        feature_rows = []
        enriched_rows = []
        for item in candidates:
            feat = self._row_features(user_id=user_id, item=item)
            feature_rows.append([feat[c] for c in feature_cols])
            enriched_rows.append((item, feat))

        feature_matrix = np.array(feature_rows, dtype=np.float32)
        ml_scores = self._score_with_ml(feature_matrix)

        ranked_results = []
        for i, (item, feat) in enumerate(enriched_rows):
            margin_boost = min(feat["margin"] / 15.0, 1.0)
            inventory_boost = min(feat["stock"] / 200.0, 1.0)
            relevance_score = float(ml_scores[i])
            final_score = (
                weights["w_relevance"] * relevance_score
                + weights["w_margin"] * margin_boost
                + weights["w_inventory"] * inventory_boost
            )

            ranked_results.append(
                {
                    "product_id": item.product_id,
                    "product_name": item.product_name,
                    "score": round(float(final_score), 4),
                    "relevance_score": round(float(relevance_score), 4),
                    "price": item.price,
                    "margin": item.margin,
                    "stock": item.stock,
                    "debug_features": {
                        "user_product_frequency": round(feat["user_product_frequency"], 4),
                        "user_product_recency": round(feat["user_product_recency"], 4),
                        "user_product_reorder_ratio": round(feat["user_product_reorder_ratio"], 4),
                        "embedding_similarity": round(feat["embedding_similarity"], 4),
                        "category_affinity": round(feat["category_affinity"], 4),
                    },
                }
            )

        return sorted(ranked_results, key=lambda x: x["score"], reverse=True)