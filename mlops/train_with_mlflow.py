import mlflow
import mlflow.pytorch
import mlflow.onnx # Added
import torch
import os
from training.train_two_tower import train
from training.train_reranker import train_reranker # Ensure this is imported

def supervised_training():
    mlflow.set_experiment("Retail_Optimization_Engine")
    
    with mlflow.start_run() as run:
        # 1. Log Hyperparameters
        mlflow.log_params({"embedding_dim": 64, "lr": 0.001})
        
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

        print(f"🚀 Full Pipeline logged to MLflow Run: {run.info.run_id}")

if __name__ == "__main__":
    supervised_training()