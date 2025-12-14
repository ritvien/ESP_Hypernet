from numpy import linalg as LA
from tqdm import tqdm
import autograd.numpy as np

def solve_CQ_feasible(f, jac_f, proj_C, proj_Qplus, x0,
                        gamma=1.0, max_iter=1000, tol=1e-6):
    
    x = proj_C(x0)
    
    x_hist = [x.copy()]
    f_hist = [f(x).copy()]      
    z_proj_hist = []  

    for k in tqdm(range(max_iter)):
        z = f(x)
        z_proj = proj_Qplus(z)          # P_{Q⁺}(f(x))
        g = jac_f(x).T @ (z - z_proj)   # ∇Φ(x)
        gamma_k = gamma / ((k + 1)**1.0001) 
        x_new = proj_C(x - gamma_k * g)
        
        err_x = LA.norm(x_new - x)
        err_f = LA.norm(z - z_proj)
                    
        x_hist.append(x.copy())
        f_hist.append(z.copy())
        z_proj_hist.append(z_proj.copy())
        
        if err_f < tol:
            print(f"\nHội tụ tại vòng lặp {k}")
            break
        x = x_new
    return x, np.array(x_hist), np.array(f_hist), np.array(z_proj_hist)