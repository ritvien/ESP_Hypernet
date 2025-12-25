from prettytable import PrettyTable
from numpy import linalg as LA
from tqdm import tqdm
import autograd.numpy as np

def fmt(arr, precision=6):
    return np.array2string(arr, precision=precision, separator=', ', suppress_small=True)

def CQ_split_acceptance(f, jac_f, proj_C, proj_Qplus, x0,
                        gamma=1.0, max_iter=1000, tol=1e-6):
    
    np.set_printoptions(precision=4, suppress=True)

    print(f"Khởi tạo: x0: {x0}")
    x = proj_C(x0)
    print(f"Chiếu lên C được: x: {x}")
    
    x_hist = [x.copy()]
    f_hist = [f(x).copy()]      
    z_proj_hist = []  
    
    table = PrettyTable()
    table.field_names = ["k", "x_new", "gamma_k", "y", "z_proj", "e_x", "e_f"]
    
    table.align["x_new"] = "l"
    table.align["y"] = "l"
    table.align["gamma_k"] = "l"
    table.align["z_proj"] = "l"

    for k in tqdm(range(max_iter)):
        z = f(x)
        z_proj = proj_Qplus(z)          # P_{Q⁺}(f(x))
        g = jac_f(x).T @ (z - z_proj)   # ∇Φ(x)
        gamma_k = gamma / ((k + 1)**1.0001) 
        x_new = proj_C(x - gamma_k * g)
        
        err_x = LA.norm(x_new - x)
        err_f = LA.norm(z - z_proj)


        if k%10==0:
            table.add_row([
                k, 
                fmt(x), 
                f"{gamma_k:.4f}", 
                fmt(z), 
                fmt(z_proj), 
                f"{err_x:.6f}", 
                f"{err_f:.6f}"
            ])
        # --------------------
                    
        x_hist.append(x.copy())
        f_hist.append(z.copy())
        z_proj_hist.append(z_proj.copy())
        
        if err_f < tol:
            print(f"\nHội tụ tại vòng lặp {k}")
            break
        x = x_new

    
    print(table)
    return x, np.array(x_hist), np.array(f_hist), np.array(z_proj_hist)