import autograd.numpy as np

def optim_Scalarization(prob, x_feasible, r, 
                    max_iter=1000, 
                    mu=0.1, 
                    expo_alpha=0.25,
                    expo_lambda=0.75,
                    init_params=1.0):
    """
    Bài toán:
        Min S(F(x))  [hàm vô hướng hóa Chebysev có trọng số]
        s.t. x in argmin S(f(x),r) := max {r_i*f_i(x)}
             x in C
    """
    
    x_curr = np.array(x_feasible).copy()
    path_x = [x_curr.copy()]
    
    P_C = prob.proj_C
    P_Qplus = prob.proj_Qplus
     


    for k in range(max_iter):
        step = k + 1
        
        fx_curr = prob.objective_func(x_curr) # F(x)
        J = prob.jacobian(x_curr)             # J(x) = ∇F(x)

        # 2. Tính Gradient Cấp thấp (v^k) -> Giải quyết ràng buộc F(x) in Q+
        proj_q = P_Qplus(fx_curr)
        r_k = fx_curr - proj_q
        
        # v^k = J^T * (F(x) - P_Q+(F(x)))
        v_k = J.T @ r_k
        
        # 3. Tính Gradient Cấp cao (w^k) -> Tối ưu hóa hàm S
        weighted_vals = r * fx_curr 
        idx_max = np.argmax(weighted_vals)
        
        # w^k = J^T * ∇ S(F(x))
        w_k = r[idx_max] * J[idx_max, :]
        
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
        
        path_x.append(x_next.copy())
        x_curr = x_next
        
        viol_q = np.linalg.norm(r_k)
        if norm_d < 1e-6 and viol_q < 1e-6:
            print(f"-> Thuật toán hội tụ sớm. Sau {k} vòng lặp")
            break


    return x_curr, path_x