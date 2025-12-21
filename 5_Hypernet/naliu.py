import numpy as np
import torch
from torch import nn, Tensor
import autograd.numpy as np1
from autograd import grad
from neural_ode import ODEF
from get_problem import Problem

def sign(g):
    if g>0:
        return torch.tensor([1])
    elif g<0:
        return torch.tensor([0])
    else:
        p = torch.rand(1)
        return p
    
class TestODEF(ODEF):
    def __init__(self, A, B, x0,s,r):
        super(TestODEF, self).__init__()
        self.A = nn.Linear(2, 2, bias=False)
        self.A.weight = nn.Parameter(A)
        self.B = nn.Linear(2, 2, bias=False)
        self.B.weight = nn.Parameter(B)
        self.x0 = nn.Parameter(x0)
        self.s = s
        self.r = r
    def forward(self, x, t):
        # print(x)
        g1, g2, g3, g4, g5, g6 = Problem.g1, Problem.g2, Problem.g3, Problem.g4, Problem.g5, Problem.g6
        g7 = Problem.g7
        J = torch.tensor(
            [
                sign(g1(x)).item(),
                sign(g2(x)).item(),
                sign(g3(x)).item(),
                sign(g4(x)).item(),
                sign(g5(x)).item(),
                sign(g6(x)).item(),
                sign(g7(x)).item(),
            ]
        )
        c = torch.prod(1-J)
        z = x.detach().cpu().numpy()
        grad_f = grad(self.s)(z,self.r)
        grad_g1 = grad(g1)(z)
        grad_g2 = grad(g2)(z)
        grad_g3 = grad(g3)(z)
        grad_g4 = grad(g4)(z)
        grad_g5 = grad(g5)(z)
        grad_g6 = grad(g6)(z)
        grad_g7 = grad(g7)(z)

        dxdt = -c*torch.tensor(grad_f) - (
            sign(g1(x))*torch.tensor(grad_g1)
            + sign(g2(x))*torch.tensor(grad_g2)
            + sign(g3(x))*torch.tensor(grad_g3)
            + sign(g4(x))*torch.tensor(grad_g4)
            + sign(g5(x))*torch.tensor(grad_g5)                                        
            + sign(g6(x))*torch.tensor(grad_g6)
            + sign(g7(x))*torch.tensor(grad_g7)
        ) 
        return dxdt