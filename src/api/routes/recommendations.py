from fastapi import APIRouter, HTTPException, Query, Body
import onnxruntime as ort
import numpy as np
import time
from typing import List, Dict, Any

from src.engine.retriever import Retriever
from src.engine.ranker import Ranker
from src.engine.session_manager import SessionManager
from src.core.config import settings

router = APIRouter()

# --- Initialize Engine Components ---
# We load the ONNX session once at startup to keep latency < 20ms
try:
    ort_session = ort.InferenceSession(settings.MODEL_PATH)
    input_name = ort_session.get_inputs()[0].name
except Exception as e:
    print(f"❌ Failed to load ONNX model: {e}")

retriever = Retriever(top_k=100) # Fetch 100 candidates for reranking
session_mgr = SessionManager()   # Handles Redis & A/B Groups

@router.post("/recommendations")
async def get_personalized_recommendations(
    user_id: int = Query(..., description="The ID of the user"),
    top_k: int = Query(10, gt=0, le=50),
    context: Dict[str, Any] = Body(default={}) # For future context-aware features
):
    """
    Full Inference Pipeline:
    1. Check Redis for cached user embedding.
    2. If missing, run ONNX model and cache result.
    3. Retrieve nearest products from Neon (Vector Search).
    4. Apply A/B group-specific business re-ranking.
    """
    start_total = time.time()
    
    try:
        # --- 1. SESSION & A/B GROUP ---
        group = await session_mgr.get_user_group(user_id)
        weights = session_mgr.get_ranking_weights(group)
        
        # --- 2. USER EMBEDDING (Step 10: Cache-Aside) ---
        # Try to skip the ONNX inference if user is active
        user_vec = await session_mgr.get_cached_embedding(user_id)
        
        inference_source = "cache"
        if not user_vec:
            inference_start = time.time()
            # Run ONNX (Step 6)
            inputs = {input_name: np.array([user_id], dtype=np.int64)}
            user_vec = ort_session.run(None, inputs)[0][0].tolist()
            
            # Save to Redis for 10 minutes
            await session_mgr.cache_embedding(user_id, user_vec)
            inference_source = "model_inference"
            inference_time = (time.time() - inference_start) * 1000
        else:
            inference_time = 0.0

        # --- 3. RETRIEVAL (Step 7: Neon pgvector) ---
        # Fetch products + price, margin, and stock
        candidates = await retriever.get_nearest_products(user_vec)
        
        if not candidates:
            return {"user_id": user_id, "recommendations": [], "status": "no_candidates"}

        # --- 4. BUSINESS RE-RANKING (Step 8: Margin Optimization) ---
        # Pass weights (0.6 rel, 0.3 margin, 0.1 stock)
        ranker = Ranker(**weights)
        final_ranked_results = ranker.rank(candidates)

        total_latency = (time.time() - start_total) * 1000

        # --- 5. RESPONSE ---
        return {
            "user_id": user_id,
            "experiment_group": group,
            "recommendations": final_ranked_results[:top_k],
            "metadata": {
                "total_latency_ms": round(total_latency, 2),
                "inference_time_ms": round(inference_time, 2),
                "inference_source": inference_source,
                "model_version": "v1.2.0-quantized"
            }
        }

    except Exception as e:
        # The GlobalExceptionHandlerMiddleware will also catch this
        raise HTTPException(status_code=500, detail=f"Engine Error: {str(e)}")