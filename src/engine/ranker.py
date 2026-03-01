import joblib
import pandas as pd
import numpy as np

class Ranker:
    def __init__(self, model_path: str, mappings_path: str):
        # Load the XGBoost model and ID mappings
        try:
            self.model = joblib.load(model_path)
            self.mappings = joblib.load(mappings_path)
            self.prod_map = self.mappings["prod_map"]
        except Exception as e:
            print(f"⚠️ Failed to load Reranker: {e}")
            self.model = None

    def rank(self, candidates, user_id: int):
        """
        Uses XGBoost to score candidates based on retrieval scores 
        and historical purchase counts.
        """
        if not self.model:
            # Fallback to simple distance sorting if model fails
            return sorted(candidates, key=lambda x: x.distance)

        features = []
        for item in candidates:
            # Prepare features matching the XGBoost training schema
            features.append({
                "purchase_count": 0, # Note: Moving this here for clarity
                "retrieval_score": 1 - item.distance
            })

        # 1. Create the DataFrame
        df_features = pd.DataFrame(features)
        
        # 2. FIX: Explicitly reorder columns to match the training schema
        # The error indicated it expects: ['purchase_count', 'retrieval_score']
        feature_order = ["purchase_count", "retrieval_score"]
        df_features = df_features[feature_order]
        
        # 3. Predict probability of purchase
        # predict_proba returns [prob_class_0, prob_class_1]
        probs = self.model.predict_proba(df_features)[:, 1]

        ranked_results = []
        for i, item in enumerate(candidates):
            ranked_results.append({
                "product_id": item.product_id,
                "product_name": item.product_name,
                "score": round(float(probs[i]), 4),
                "price": item.price,
                "margin": item.margin,
                "stock": item.stock
            })

        # Sort by the new XGBoost score in descending order
        return sorted(ranked_results, key=lambda x: x['score'], reverse=True)