import mlflow
import mlflow.pytorch
import mlflow.onnx # Added
import torch
import os
import json
from training.train_two_tower import train
from training.train_reranker import train_reranker # Ensure this is imported
from training.offline_eval import evaluate

def supervised_training():
    mlflow.set_experiment("Retail_Optimization_Engine")
    
    with mlflow.start_run() as run:
        # 1. Log Hyperparameters
        mlflow.log_params({
            "embedding_dim": 64,
            "lr": 0.001,
            "reranker_objective": "rank:ndcg",
            "candidate_pool": 200,
            "business_weighting_policy": "relevance_first",
        })
        
        # 2. Train Two-Tower (Neural Retrieval)
        model, metrics = train() 
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, artifact_path="retrieval_model")
        
        # 3. Train Reranker (which now generates ONNX)
        train_reranker()
        
        # 4. Log the Reranker Artifacts
        if os.path.exists("models/reranker.onnx"):
            mlflow.log_artifact("models/reranker.onnx", artifact_path="reranker_production")
            print("✅ ONNX Reranker logged as production artifact.")
        
        if os.path.exists("models/reranker_xgb.pkl"):
            mlflow.log_artifact("models/reranker_xgb.pkl", artifact_path="reranker_debug")

        if os.path.exists("models/reranker_features.pkl"):
            mlflow.log_artifact("models/reranker_features.pkl", artifact_path="reranker_debug")

        # 5. Track offline comparison metrics for retrieval vs rerank stages
        evaluate()
        if os.path.exists("mlops/reports/offline_ranking_metrics.json"):
            with open("mlops/reports/offline_ranking_metrics.json", "r", encoding="utf-8") as f:
                ranking_metrics = json.load(f)

            mlflow.log_metrics(
                {
                    "retrieval_only_ndcg10": ranking_metrics["retrieval_only"]["ndcg@10"],
                    "heuristic_rerank_ndcg10": ranking_metrics["retrieval_plus_heuristic_rerank"]["ndcg@10"],
                    "ml_rerank_ndcg10": ranking_metrics["retrieval_plus_ml_rerank"]["ndcg@10"],
                    "retrieval_only_precision10": ranking_metrics["retrieval_only"]["precision@10"],
                    "ml_rerank_precision10": ranking_metrics["retrieval_plus_ml_rerank"]["precision@10"],
                }
            )
            mlflow.log_artifact("mlops/reports/offline_ranking_metrics.json", artifact_path="evaluation")

        print(f"🚀 Full Pipeline logged to MLflow Run: {run.info.run_id}")

if __name__ == "__main__":
    supervised_training()