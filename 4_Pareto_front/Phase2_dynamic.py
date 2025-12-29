import autograd.numpy as np
from prettytable import PrettyTable

def optim_Scalarization(prob, x_feasible, r, z_star,
                                     max_iter=1000, 
                                     mu=1e-6,                 # Ngưỡng normalization nhỏ
                                     init_params=1.0,
                                     # --- Tham số mũ cho thuật toán 1-A ---
                                     expo_alpha=0.25,          # Giảm: Trọng số hàm mục tiêu (a)
                                     expo_lambda=0.75,         # Giảm: Bước nhảy (l), 0.5 < l <= 1
                                     expo_beta=0.75,           # Tăng: Phạt tập nguồn C (b_C) < l
                                     expo_gamma=0.35,         # Tăng: Phạt tập đích Q+ (b_Q) < b_C < l
                                     verbose=True
                       ):
    """
    Algorithm 1-A: Balanced Penalty Method cho bài toán ESP.
    
    Công thức cập nhật:
        d^k = gamma_k * v^k + alpha_k * w^k + beta_k * z^k
        x^{k+1} = x^k - (lambda_k / eta_k) * d^k
    
    Trong đó:
        v^k: Gradient vi phạm tập Q+ (SFP)
        w^k: Subgradient hàm mục tiêu S
        z^k: Gradient vi phạm tập C
    """
    
    x_curr = np.array(x_feasible, dtype=np.float64).copy()
    path_x = [x_curr.copy()]
    
    P_C = prob.proj_C
    P_Qplus = prob.proj_Qplus
    obj_val_prev = float('inf')
    
    if verbose:
        table = PrettyTable()
        table.field_names = ["k", "Alpha", "Beta(C)", "Gamma(Q)", "Lambda", "Step", "S(F(x))", "Gap C", "Gap Q+"]
        table.float_format = ".5"
        table.align = "r" 

    print(f"Params: alpha=k^-{expo_alpha}, lambda=k^-{expo_lambda}, beta=k^{expo_beta}, gamma=k^{expo_gamma}")

    for k in range(max_iter):
        step = k + 1
        
        # 1. Tính toán cơ bản
        fx_curr = prob.objective_func(x_curr) 
        J = prob.jacobian(x_curr)             

        # 2. Tính Gradient Cấp thấp v^k (Ràng buộc Q+)
        proj_q = P_Qplus(fx_curr)
        gap_vec_q = fx_curr - proj_q        
        v_k = J.T @ gap_vec_q
        
        # 3. Tính Gradient Cấp cao w^k (Hàm mục tiêu S)
        weighted_vals = r * (fx_curr - z_star) 
        idx_max = np.argmax(weighted_vals)
        w_k = r[idx_max] * J[idx_max, :]
        
        current_obj_val = weighted_vals[idx_max]
        
        # 4. Tính Gradient Phạt tập C z^k (Ràng buộc C)
        proj_c_val = P_C(x_curr)
        z_k = x_curr - proj_c_val  
        
        # 5. Cập nhật tham số động (Theo giả thiết hội tụ)
        alpha_k = init_params / (step ** expo_alpha)   # Giảm dần
        lambda_k = init_params / (step ** expo_lambda) # Giảm dần
        beta_k = init_params * (step ** expo_beta)     # Tăng dần (Kéo về C)
        gamma_k = init_params * (step ** expo_gamma)   # Tăng dần (Kéo về Q+)
        
        # 6. Tổng hợp hướng di chuyển d^k
        # Algorithm 1-A: Có hệ số gamma_k cho v_k
        d_k = (gamma_k * v_k) + (alpha_k * w_k) + (beta_k * z_k)
        
        # 7. Chuẩn hóa và Cập nhật
        norm_d = np.linalg.norm(d_k)
        eta_k = max(mu, norm_d)
        
        actual_step_size = lambda_k / eta_k
        
        # Di chuyển ngược hướng Gradient tổng hợp
        x_next = x_curr - actual_step_size * d_k
        
        path_x.append(x_next.copy())
        x_curr = x_next
        
        # --- Đánh giá vi phạm và mục tiêu ---
        val_S = np.linalg.norm(fx_curr - z_star) # Chỉ để hiển thị
        viol_q = np.linalg.norm(gap_vec_q)       # Norm sai số tập Q+
        viol_c = np.linalg.norm(z_k)             # Norm sai số tập C
        delta_obj = abs(current_obj_val - obj_val_prev)
            
        if verbose:
            if k % 250 == 0 or k == max_iter - 1:
                table.add_row([k, alpha_k, beta_k, gamma_k, lambda_k, actual_step_size, val_S, viol_c, viol_q])
        
        # --- Điều kiện dừng ---
        if viol_q < 1e-5 and viol_c < 1e-5 and delta_obj < 1e-5:
            print(f"-> Thuật toán hội tụ sớm tại k={k}.")
            break
        obj_val_prev = current_obj_val
            
    if step == max_iter:
        print(f"!! Max_iter. Delta: {delta_obj:.6f}, Gap C: {viol_c:.6f}, Gap Q: {viol_q:.6f}")
        
    if verbose:
        print(table)
        final_fx = prob.objective_func(x_curr)
        final_viol_q = np.linalg.norm(final_fx - P_Qplus(final_fx))
        print(f"-> Kết quả cuối cùng: {x_curr}")
        print(f"-> S(F(x)) approx distance: {np.linalg.norm(final_fx - z_star):.5f}")

    return x_curr, path_x