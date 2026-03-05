import joblib
import numpy as np
import onnxruntime as ort

class Ranker:
    def __init__(self, model_path: str, mappings_path: str):
        try:
            # Load ONNX session instead of XGBoost pickle
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name
            
            # Mappings are still in joblib format
            self.mappings = joblib.load(mappings_path)
            self.prod_map = self.mappings["prod_map"]
        except Exception as e:
            print(f"⚠️ Failed to load ONNX Reranker: {e}")
            self.session = None

    def rank(self, candidates, user_id: int, weights: dict = None):
        """
        Inference using ONNX runtime. 
        Replaces XGBoost predict_proba with onnx session run.
        """
        if not self.session:
            # Fallback to retrieval distance if model fails to load
            return sorted(candidates, key=lambda x: x.distance)

        # 1. Prepare Features for ONNX (Strict NumPy array [Rows, 2])
        # Feature 0: purchase_count | Feature 1: retrieval_score
        features = []
        for item in candidates:
            features.append([
                float(getattr(item, 'purchase_count', 0)),
                float(1 - item.distance)
            ])
        
        input_data = np.array(features, dtype=np.float32)

        # 2. Get ML Purchase Probability from ONNX
        # ONNX returns a list: [labels, probabilities]
        # probabilities is a list of dictionaries like [{0: 0.9, 1: 0.1}, ...]
        outputs = self.session.run(None, {self.input_name: input_data})
        
        # Extract probability of class 1 (purchase)
        raw_probs = outputs[1]
        probs = [p[1] for p in raw_probs]

        # 3. Apply Business Logic Weights
        if not weights:
            weights = {"w_relevance": 1.0, "w_margin": 0.0, "w_inventory": 0.0}

        ranked_results = []
        for i, item in enumerate(candidates):
            # Normalize margin (assuming max margin is ~15 for scaling)
            norm_margin = min(item.margin / 15.0, 1.0) 
            
            # Final Business Score Formula
            # Using probs[i] which we got from ONNX session
            final_score = (
                weights["w_relevance"] * float(probs[i]) +
                weights["w_margin"] * norm_margin
            )

            ranked_results.append({
                "product_id": item.product_id,
                "product_name": item.product_name,
                "score": round(final_score, 4),
                "price": item.price,
                "margin": item.margin,
                "stock": item.stock
            })

        # Return list sorted by the calculated final_score
        return sorted(ranked_results, key=lambda x: x['score'], reverse=True)