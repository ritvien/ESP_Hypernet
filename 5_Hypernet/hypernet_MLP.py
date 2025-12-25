import torch
from torch import nn
import torch.nn.functional as F

class Hypernet_MLP(nn.Module):
    def __init__(self, ray_hidden_dim=32, out_dim=2, target_hidden_dim=15, n_hidden=1, n_tasks=2):
        """
        Phiên bản MLP thay thế cho Hypernet_trans.
        Giữ nguyên các tham số đầu vào để tương thích code cũ.
        """
        super().__init__()
        self.n_tasks = n_tasks
        
        # Kiến trúc MLP đơn giản:
        self.net = nn.Sequential(
            nn.Linear(n_tasks, ray_hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            
            nn.Linear(ray_hidden_dim, out_dim)
        )

    def forward(self, ray):
        """
        Input ray: Tensor shape (Batch_Size, n_tasks)
        Output x: Tensor shape (1, Batch_Size, out_dim) 
                  (Shape (1,...) để khớp với output cũ của Transformer sau khi unsqueeze)
        """
        
        x = self.net(ray) # Output shape: (Batch_Size, out_dim)
        
        x = x.unsqueeze(0)
        
        return x