import torch
from torch import nn
import torch.nn.functional as F

class Hypernet_trans(nn.Module):
    def __init__(self, ray_hidden_dim=32, out_dim=2, target_hidden_dim=15, n_hidden=1, n_tasks=2):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_tasks = n_tasks

        # Các layer giữ nguyên
        if self.n_tasks == 2:
            self.embedding_layer1 = nn.Sequential(nn.Linear(1, ray_hidden_dim), nn.ReLU(inplace=True))
            self.embedding_layer2 = nn.Sequential(nn.Linear(1, ray_hidden_dim), nn.ReLU(inplace=True))
        else:
            self.embedding_layer1 = nn.Sequential(nn.Linear(1, ray_hidden_dim), nn.ReLU(inplace=True))
            self.embedding_layer2 = nn.Sequential(nn.Linear(1, ray_hidden_dim), nn.ReLU(inplace=True))
            self.embedding_layer3 = nn.Sequential(nn.Linear(1, ray_hidden_dim), nn.ReLU(inplace=True))
            
        self.output_layer = nn.Linear(ray_hidden_dim, out_dim)
        self.attention = nn.MultiheadAttention(embed_dim=ray_hidden_dim, num_heads=2)
        self.ffn1 = nn.Linear(ray_hidden_dim, ray_hidden_dim)
        self.ffn2 = nn.Linear(ray_hidden_dim, ray_hidden_dim)

    def forward(self, ray):
        """
        Input ray: (Batch_Size, n_tasks) -> VD: (20, 2)
        Output x: (Batch_Size, out_dim) -> VD: (20, 2)
        """
        
        if self.n_tasks == 2: 
            emb1 = self.embedding_layer1(ray[:, 0].unsqueeze(1)) 
            emb2 = self.embedding_layer2(ray[:, 1].unsqueeze(1))
            
            x = torch.stack((emb1, emb2), dim=0) 
        else:
            x = torch.stack(
                (
                    self.embedding_layer1(ray[:, 0].unsqueeze(1)),
                    self.embedding_layer2(ray[:, 1].unsqueeze(1)),
                    self.embedding_layer3(ray[:, 2].unsqueeze(1))
                ), dim=0
            )
        
        # --- Transformer Block (Seq_len, Batch, Dim) ---
        x_res = x
        x, _ = self.attention(x, x, x)
        x = x + x_res
        
        x_res = x
        x = self.ffn1(x)
        x = F.relu(x)
        x = self.ffn2(x)
        x = x + x_res
        
        # --- Output Layer ---
        # x: (Seq_len, Batch, Hidden) -> (Seq_len, Batch, Out_dim)
        x = self.output_layer(x)
        
        # --- Aggregation ---
        # Kết quả: (Batch, Out_dim)
        x = torch.mean(x, dim=0) 
        
        
        return x