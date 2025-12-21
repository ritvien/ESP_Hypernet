import autograd.numpy as np

def optim_Universal(prob, x_feasible, r=None, 
                    target_dim=None,    
                    mode='min',        
                    max_iter=1000, 
                    mu=0.1, 
                    expo_alpha=0.25,
                    expo_lambda=0.75,
                    init_params=1.0):
    """
    Giải bài toán ESP linh hoạt:
    
    """
    
    x_curr = np.array(x_feasible).copy()
    path_x = [x_curr.copy()]
    
    P_C = prob.proj_C
    P_Qplus = prob.proj_Qplus
    
    # Nếu chạy Chebyshev thì cần r, nếu không thì r có thể là None
    if target_dim is None and r is None:
        raise ValueError("Nếu target_dim là None (chạy Chebyshev), bạn phải cung cấp vector r")

    for k in range(max_iter):
        step = k + 1
        
        fx_curr = prob.objective_func(x_curr) # F(x)
        J = prob.jacobian(x_curr)             # J(x) = ∇F(x)

        # ---------------------------------------------------------
        # 2. Tính Gradient Cấp thấp (v^k) -> Kéo về tập Q+ (GIỮ NGUYÊN)
        # ---------------------------------------------------------
        proj_q = P_Qplus(fx_curr)
        r_k = fx_curr - proj_q
        v_k = J.T @ r_k
        
        # ---------------------------------------------------------
        # 3. Tính Gradient Cấp cao (w^k) -> XỬ LÝ LINH HOẠT
        # ---------------------------------------------------------
        
        if target_dim is not None:
            # Chỉ quan tâm đến dòng thứ `target_dim` của Jacobian
            
            grad_f_i = J[target_dim, :] # Lấy hàng tương ứng
            
            if mode == 'min':
                # Bài toán Min: w^k là hướng gradient dương (để đi ngược lại trong bước cập nhật)
                # Vì công thức cập nhật là: x - step * d_k
                w_k = grad_f_i
            elif mode == 'max':
                # Bài toán Max f(x) <=> Min -f(x)
                # Gradient của -f(x) là -Gradient f(x)
                w_k = -grad_f_i
            else:
                raise ValueError("mode phải là 'min' hoặc 'max'")
                
     
        alpha_k = init_params / (step ** expo_alpha)
        lambda_k = init_params / (step ** expo_lambda)
        
        d_k = v_k + alpha_k * w_k
        
        norm_d = np.linalg.norm(d_k)
        eta_k = max(mu, norm_d)
        
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