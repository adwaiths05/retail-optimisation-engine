import joblib
import pandas as pd
import numpy as np

class Ranker:
    def __init__(self, model_path: str, mappings_path: str):
        try:
            self.model = joblib.load(model_path)
            self.mappings = joblib.load(mappings_path)
            self.prod_map = self.mappings["prod_map"]
        except Exception as e:
            print(f"⚠️ Failed to load Reranker: {e}")
            self.model = None

    def rank(self, candidates, user_id: int, weights: dict = None):
        """
        Combines XGBoost purchase probability with business weights 
        (margin/inventory) based on the experiment group.
        """
        if not self.model:
            return sorted(candidates, key=lambda x: x.distance)

        # 1. Prepare Features for XGBoost
        features = []
        for item in candidates:
            features.append({
                "purchase_count": getattr(item, 'purchase_count', 0), # Pull actual counts if available
                "retrieval_score": 1 - item.distance
            })

        df_features = pd.DataFrame(features)[["purchase_count", "retrieval_score"]]
        
        # 2. Get ML Purchase Probability
        probs = self.model.predict_proba(df_features)[:, 1]

        # 3. Apply Business Logic Weights
        if not weights:
            # Default weights if none provided
            weights = {"w_relevance": 1.0, "w_margin": 0.0, "w_inventory": 0.0}

        ranked_results = []
        for i, item in enumerate(candidates):
            # Normalize margin (assuming max margin is ~15 for scaling)
            norm_margin = min(item.margin / 15.0, 1.0) 
            
            # Final Business Score Formula
            final_score = (
                weights["w_relevance"] * float(probs[i]) +
                weights["w_margin"] * norm_margin
            )

            ranked_results.append({
                "product_id": item.product_id,
                "product_name": item.product_name,
                "score": round(final_score, 4), # This will no longer be 0
                "price": item.price,
                "margin": item.margin,
                "stock": item.stock
            })

        return sorted(ranked_results, key=lambda x: x['score'], reverse=True)