import autograd.numpy as np
from prettytable import PrettyTable

def optimize_phase2(prob, x_feasible, u_star, 
                    max_iter=1000, 
                    mu=0.1, 
                    expo_alpha=0.25,
                    expo_lambda=0.75,
                    init_params=1.0, 
                    verbose=True):
    """
    Bài toán:
        Min S(F(x))  [tương ứng với việc tối thiểu khoảng cách tới u_star]
        s.t. x in argmin ||F(x) - P_Q+(F(x))||^2  [x thỏa mãn F(x) in Q+]
             x in C
    """
    
    x_curr = np.array(x_feasible).copy()
    path_x = [x_curr.copy()]
    
    P_C = prob.proj_C
    P_Qplus = prob.proj_Qplus
    
    if verbose:
        table = PrettyTable()
        table.field_names = ["k", "Alpha_k", "Step_len", "x_curr", "S(F(x))", "Gap Q+"]
        table.float_format = ".4"
        table.align = "r" 


    for k in range(max_iter):
        step = k + 1
        
        fx_curr = prob.objective_func(x_curr) # F(x)
        J = prob.jacobian(x_curr)             # J(x) = ∇F(x)

        # 2. Tính Gradient Cấp thấp (v^k) -> Giải quyết ràng buộc F(x) in Q+
        # r^k = F(x) - P_Q+(F(x))
        proj_q = P_Qplus(fx_curr)
        gap_vector = fx_curr - proj_q
        
        # v^k = J^T * (F(x) - P_Q+(F(x)))
        v_k = J.T @ gap_vector
        
        # 3. Tính Gradient Cấp cao (w^k) -> Tối ưu hóa hàm S
        grad_S = fx_curr - u_star
        
        # w^k = J^T * ∇ S(F(x))
        w_k = J.T @ grad_S
        
        # 4. Cập nhật tham số
        alpha_k = init_params / (step ** expo_alpha)
        lambda_k = init_params / (step ** expo_lambda)
        
        # 5. Tổng hợp hướng di chuyển d^k
        d_k = v_k + alpha_k * w_k
        
        # 6. Tính hệ số chuẩn hóa eta_k
        norm_d = np.linalg.norm(d_k)
        eta_k = max(mu, norm_d)
        
        # 7. Bước cập nhật và Chiếu lên C
        step_size = lambda_k / eta_k
        x_temp = x_curr - step_size * d_k
        x_next = P_C(x_temp)
        
        # --- Lưu và Log ---
        val_S = np.linalg.norm(fx_curr - u_star) 
        viol_q = np.linalg.norm(gap_vector)      
        
        if verbose:
            if k % 50 == 0 or k == max_iter - 1:
                x_str = np.array2string(x_curr, precision=3, separator=',')
                table.add_row([k, f"{alpha_k:.4f}", f"{step_size:.4f}", x_str, val_S, viol_q])
        
        path_x.append(x_next.copy())
        x_curr = x_next
        
        # Điều kiện dừng
        if norm_d < 1e-6 and viol_q < 1e-6:
            print(f"-> Thuật toán hội tụ sớm. Sau {k} vòng lặp")
            break

    if verbose:
        print(table)
        final_fx = prob.objective_func(x_curr)
        final_viol_q = np.linalg.norm(final_fx - P_Qplus(final_fx))
        print(f"-> Kết quả cuối cùng Phase 2: {x_curr}")
        print(f"-> S(F(x)): {np.linalg.norm(final_fx - u_star):.4f}")
        print(f"-> Sai số ràng buộc Q+ (Gap): {final_viol_q:.6f}")

    return x_curr, path_x