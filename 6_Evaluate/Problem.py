import numpy as np
class Problem():
    def __init__(self, f, jac_f, dim_x, dim_y, proj_C, proj_Qplus):
        self.f = f
        self.jac_f = jac_f
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.proj_C = proj_C
        self.proj_Qplus = proj_Qplus
    
    def objective_func(self, x):
        vals = [func(x) for func in self.f]
        return np.concatenate(vals)
    
    def jacobian(self, x):
        jacs = [func(x) for func in self.jac_f]
        return np.vstack(jacs)
    
    
