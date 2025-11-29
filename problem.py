import numpy as np
class Problem():
    def __init__(self, f, jac_f, C, Q, dim_x, dim_y, proj_C, proj_Qplus):
        self.f = f
        self.jac_f = jac_f
        self.C = C
        self.Q = Q
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.proj_C = proj_C
        self.proj_Qplus = proj_Qplus
    
    def objective_func(self, x):
        return np.array([func(x) for func in self.f])
    
    def jacobian(self, x):
        return np.array([func(x) for func in self.jac_f])