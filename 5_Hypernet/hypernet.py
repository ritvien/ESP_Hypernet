import torch
from torch import nn
import torch.nn.functional as F


class Hypernet_trans(nn.Module):
    def __init__(self, ray_hidden_dim=32, out_dim=2, target_hidden_dim=15, n_hidden=1, n_tasks=2):
        super().__init__()
        self.n_hidden = n_hidden
        self.n_tasks = n_tasks

        if self.n_tasks == 2:
            self.embedding_layer1 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
            self.embedding_layer2 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
        else:
            self.embedding_layer1 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
            self.embedding_layer2 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
            self.embedding_layer3 =  nn.Sequential(nn.Linear(1, ray_hidden_dim),nn.ReLU(inplace=True))
            
        self.output_layer =  nn.Linear(ray_hidden_dim, out_dim)
        self.attention = nn.MultiheadAttention(embed_dim=ray_hidden_dim, num_heads=2)
        self.ffn1 = nn.Linear(ray_hidden_dim,ray_hidden_dim)
        self.ffn2 = nn.Linear(ray_hidden_dim, ray_hidden_dim)
        # self.feature = ode

    def forward(self, ray):
        if self.n_tasks == 2: 
            x = torch.stack((self.embedding_layer1(ray[:,0]),self.embedding_layer2(ray[:,1])))
        else:
            x = torch.stack(
                (
                    self.embedding_layer1(ray[:,0].unsqueeze(1)),
                    self.embedding_layer2(ray[:,1].unsqueeze(1)),
                    self.embedding_layer3(ray[:,2].unsqueeze(1))
                )
            )
        x_ = x
                
        x,_ = self.attention(x,x,x)
        x = x + x_
        x_ = x
        x = self.ffn1(x)
        x = F.relu(x)
        x = self.ffn2(x)
        x = x + x_
        x = self.output_layer(x)
#         print(x.shape)
        x = torch.mean(x,dim=0) 
        # print(x)
        # x = x.unsqueeze(0)
        x = x.unsqueeze(0)
#         print(t.shape,x.shape)
        # print(x.shape)
        # x = self.feature(x,t,return_whole_sequence=True) 
#         print(x)
        return x