import mlflow
import mlflow.pytorch
import torch
from training.train_two_tower import train # Your existing training function

def supervised_training():
    mlflow.set_experiment("Retail_Optimization_Engine")
    
    with mlflow.start_run() as run:
        # Log Hyperparameters
        mlflow.log_params({"embedding_dim": 64, "lr": 0.001})
        
        # Run your training logic
        model, metrics = train() 
        
        # Log Metrics & Model
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(
            model, 
            artifact_path="model",
            registered_model_name="Retail_TwoTower_Model"
        )
        print(f"🚀 Model logged to MLflow Run: {run.info.run_id}")

if __name__ == "__main__":
    supervised_training()