import autograd.numpy as np
from prettytable import PrettyTable

def optim_Scalarization(prob, x_feasible, r, z_star,
                                     max_iter=1000, 
                                     mu=0.1, 
                                     expo_alpha=0.25,
                                     expo_lambda=0.75,
                                     expo_beta=0.85,
                                     init_params=1.0,
                                     verbose=True
                       ):
    """
    Bài toán: Giải quyết bài toán hai cấp bằng phương pháp Neurodynamic/Penalty
    (Không dùng phép chiếu cứng P_C ở bước cập nhật cuối cùng)
    
    Args:
        beta_param (float): Hệ số phạt để kéo x về tập C (nên chọn giá trị dương lớn, vd: 10, 100...)
    """
    
    x_curr = np.array(x_feasible).copy()
    path_x = [x_curr.copy()]
    
    P_C = prob.proj_C
    P_Qplus = prob.proj_Qplus
    if verbose:
        table = PrettyTable()
        table.field_names = ["k", "Alpha_k","Lambda", "Eta", "Step_len", "x_curr", "S(F(x))","Gap C", "Gap Q+"]
        table.float_format = ".4"
        table.align = "r" 

    for k in range(max_iter):
        step = k + 1
        
        fx_curr = prob.objective_func(x_curr) 
        J = prob.jacobian(x_curr)             

        # --- 1. Tính Gradient Cấp thấp (v^k) : Giải quyết ràng buộc F(x) in Q+ ---
        proj_q = P_Qplus(fx_curr)
        r_k = fx_curr - proj_q        # Residual vector for Q+
        v_k = J.T @ r_k
        
        # --- 2. Tính Gradient Cấp cao (w^k) : Tối ưu hóa hàm S ---
        weighted_vals = r * (fx_curr - z_star) 
        idx_max = np.argmax(weighted_vals)
        w_k = r[idx_max] * J[idx_max, :]
        
        # --- 3. [MỚI] Tính Gradient Phạt tập C (z^k) : Kéo x về C ---
        # Đạo hàm của 0.5 * ||x - P_C(x)||^2 là (x - P_C(x))
        proj_c_val = P_C(x_curr)
        grad_penalty_C = x_curr - proj_c_val  
        
        # --- 4. Cập nhật tham số ---
        alpha_k = init_params / (step ** expo_alpha)
        lambda_k = init_params / (step ** expo_lambda)
        beta_k = init_params * (step ** expo_beta)
        
        # --- 5. Tổng hợp hướng di chuyển d^k (Total Gradient) ---
        # d^k = Gradient_SFP + alpha * Gradient_Objective + beta * Gradient_ConstraintC
        d_k = v_k + (alpha_k * w_k) + (beta_k * grad_penalty_C)
        
        # --- 6. Tính hệ số chuẩn hóa eta_k ---
        norm_d = np.linalg.norm(d_k)
        eta_k = max(mu, norm_d)
        # --- 7. Bước cập nhật ---
        step_size = lambda_k / eta_k
        
        # Di chuyển ngược hướng Gradient tổng hợp
        x_next = x_curr - step_size * d_k
        
        path_x.append(x_next.copy())
        x_curr = x_next
        
        # --- Kiểm tra hội tụ ---
        val_S = np.linalg.norm(fx_curr - z_star) 
        viol_q = np.linalg.norm(r_k)            # Vi phạm tập Q+ (ảnh)
        viol_c = np.linalg.norm(grad_penalty_C) # Vi phạm tập C (nguồn) - [MỚI]
        if verbose:
            if k % 50 == 0 or k == max_iter - 1:
                x_str = np.array2string(x_curr, precision=3, separator=',')
                table.add_row([k, f"{alpha_k:.4f}",f"{lambda_k:.4f}", f"{eta_k:.4f}", f"{step_size:.4f}", x_str, val_S, viol_c, viol_q])
        
        # Điều kiện dừng: Gradient nhỏ VÀ các vi phạm ràng buộc đều nhỏ
        if norm_d < 1e-6 and viol_q < 1e-6 and viol_c < 1e-6:
            print(f"-> Thuật toán hội tụ sớm. Sau {k} vòng lặp")
            break
    if verbose:
        print(table)
        final_fx = prob.objective_func(x_curr)
        final_viol_q = np.linalg.norm(final_fx - P_Qplus(final_fx))
        print(f"-> Kết quả cuối cùng Phase 2: {x_curr}")
        print(f"-> S(F(x)): {np.linalg.norm(final_fx - z_star):.4f}")
        print(f"-> Sai số ràng buộc Q+ (Gap): {final_viol_q:.6f}")

    return x_curr, path_x