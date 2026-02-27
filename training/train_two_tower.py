import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from training.dataset import RetailDataset
from training.model import TwoTowerModel
import os

def train():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training with Negative Sampling on: {torch.cuda.get_device_name(0)}")

    # 2. Load and Split Data
    # Path assumes you are running from the root 'retail-opt-engine' folder
    ds = RetailDataset("./data/raw/order_products__prior.csv", "./data/raw/products.csv")
    
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = random_split(ds, [train_size, val_size])

    # Increased batch size for RTX 4060 efficiency
    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=4096, shuffle=False)

    # 3. Initialize Model
    num_users = int(ds.interactions['user_id'].max() + 1)
    num_products = ds.num_products
    model = TwoTowerModel(num_users, num_products).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss() 

    # 4. Training Loop
    epochs = 3 
    print(f"📉 Starting training (approx. 7,900 batches per epoch)...")

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for i, batch in enumerate(train_loader):
            u = batch['user_id'].to(device)
            p_pos = batch['product_idx'].to(device)
            
            # --- NEGATIVE SAMPLING MAGIC ---
            # Create negative samples by shuffling the products in this batch
            p_neg = p_pos[torch.randperm(p_pos.size(0))]
            
            optimizer.zero_grad()

            # Get scores for both
            pos_scores = model(u, p_pos)
            neg_scores = model(u, p_neg)
            
            # Combine into one prediction vector and one target vector
            logits = torch.cat([pos_scores, neg_scores])
            targets = torch.cat([
                torch.ones(u.size(0)), # Positives = 1
                torch.zeros(u.size(0)) # Negatives = 0
            ]).to(device)
            
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

            if i % 500 == 0:
                print(f"Epoch {epoch+1} | Batch {i} | Loss: {loss.item():.4f}")

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                u, p = batch['user_id'].to(device), batch['product_idx'].to(device)
                p_neg = p[torch.randperm(p.size(0))]
                
                logits = torch.cat([model(u, p), model(u, p_neg)])
                targets = torch.cat([torch.ones(u.size(0)), torch.zeros(u.size(0))]).to(device)
                val_loss += criterion(logits, targets).item()

        avg_train = total_train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"✅ Epoch {epoch+1} Results -> Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

    # --- INTERACTIVE SAVE ---
    print("\n" + "="*30)
    print("🎯 TRAINING FINISHED")
    print("="*30)
    
    confirm = input("The model has finished learning. Save weights to 'models/two_tower_best.pth'? (y/n): ").lower()

    if confirm == 'y':
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/two_tower_best.pth")
        print("💾 Model saved! Ready for vector generation.")
    else:
        print("❌ Model discarded.")

if __name__ == "__main__":
    train()