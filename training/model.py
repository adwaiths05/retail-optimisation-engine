import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, num_users, num_products, embedding_dim=64):
        super(TwoTowerModel, self).__init__()
        
        # User Tower
        self.user_embedding = nn.Embedding(num_users + 1, embedding_dim)
        self.user_fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )
        
        # Product Tower
        self.product_embedding = nn.Embedding(num_products + 1, embedding_dim)
        self.product_fc = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def forward(self, user_ids, product_ids):
        u_vec = self.user_fc(self.user_embedding(user_ids))
        p_vec = self.product_fc(self.product_embedding(product_ids))
        # Dot product similarity
        return (u_vec * p_vec).sum(dim=1)