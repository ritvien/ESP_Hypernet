import torch
import torch.optim as optim
import numpy as np
import math
import time

def evaluate_objectives_single(functions, x_tensor):
    vals = []
    for func in functions:
        val = func(x_tensor)
        if not torch.is_tensor(val):
            val = torch.tensor(val, dtype=torch.float32)
        vals.append(val)
    return torch.stack(vals).reshape(-1)

def train_hypernet(hypernet, prob, z_star, 
                   num_epochs=1000, 
                   lr=1e-3, 
                   num_partitions=100, 
                   lr_step_size=300, 
                   lr_gamma=0.5,
                   # --- THAM SỐ THUẬT TOÁN 2-A (Monotonic Penalty) ---
                   beta_C_0=1.0,    # Giá trị khởi tạo cho C
                   beta_C_max=1000.0, # Giá trị tối đa cho C
                   rho_C=1.01,      # Tỷ lệ tăng cho C 
                   
                   beta_Q_0=1.0,    # Giá trị khởi tạo cho Q 
                   beta_Q_max=1000.0, # Giá trị tối đa cho Q
                   rho_Q=1.01,      # Tỷ lệ tăng cho Q
                   verbose=True): 
    
    # 1. Khởi tạo
    optimizer = optim.Adam(hypernet.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size, gamma=lr_gamma)
    
    # Đảm bảo z_star chuẩn shape
    z_star_tensor = torch.tensor(z_star, dtype=torch.float32).view(1, -1)
    
    # Khởi tạo hệ số phạt
    beta_C = beta_C_0
    beta_Q = beta_Q_0
    
    angle_step = (math.pi / 2) / num_partitions
    
    if verbose:
        print(f"=== TRAIN HYPERNET (Algorithm 2-A: Monotonic Penalty) ===")
        print(f"Constraint C: Start {beta_C_0} -> Max {beta_C_max} (Rate {rho_C})")
        print(f"Constraint Q: Start {beta_Q_0} -> Max {beta_Q_max} (Rate {rho_Q})")
    
    for epoch in range(num_epochs):
        hypernet.train()
        optimizer.zero_grad()
        
        # --------------------------------------------------------
        # 1. Lấy mẫu phân tầng (Stratified Sampling)
        # --------------------------------------------------------
        starts = torch.arange(num_partitions) * angle_step
        noise = torch.rand(num_partitions) * angle_step
        thetas = starts + noise 
        
        r_batch_np = np.stack([np.cos(thetas.numpy()), np.sin(thetas.numpy())], axis=1)
        r_tensor_batch = torch.tensor(r_batch_np, dtype=torch.float32)
        
        # --------------------------------------------------------
        # 2. Lan truyền xuôi & Tính Loss 
        # --------------------------------------------------------
        
        # Forward pass 
        x_pred_list = []
        for i in range(num_partitions):
            r_single = r_tensor_batch[i].unsqueeze(0)
            x_single = hypernet(r_single)
            x_pred_list.append(x_single)
            
        x_vec_batch = torch.cat(x_pred_list, dim=0) # Batch output
        
        # Loop tính Loss thành phần 
        x_np_batch = x_vec_batch.detach().cpu().numpy()
        loss_C_list = []
        loss_Q_list = []
        y_pred_list = []
        
        for i in range(num_partitions):
            x_i_tensor = x_vec_batch[i].reshape(-1) 
            x_i_np = x_i_tensor.detach().cpu().numpy()
            
            # a. Loss C: ||x - P_C(x)||^2
            x_proj_i_np = prob.proj_C(x_i_np)
            x_proj_i_tensor = torch.tensor(x_proj_i_np, dtype=torch.float32)
            loss_C_list.append(torch.sum((x_i_tensor - x_proj_i_tensor)**2))
            
            # b. F(x)
            y_pred_i = evaluate_objectives_single(prob.f, x_i_tensor)
            y_pred_list.append(y_pred_i)
            
            # c. Loss Q: ||y - P_Q+(y)||^2
            y_i_np = y_pred_i.detach().cpu().numpy()
            y_proj_i_np = prob.proj_Qplus(y_i_np)
            y_proj_i_tensor = torch.tensor(y_proj_i_np, dtype=torch.float32)
            loss_Q_list.append(torch.sum((y_pred_i - y_proj_i_tensor)**2))
        
        # Tính trung bình Loss (Mean L_C, Mean L_Q)
        y_pred_batch = torch.stack(y_pred_list)
        L_bar_C = torch.mean(torch.stack(loss_C_list))
        L_bar_Q = torch.mean(torch.stack(loss_Q_list))
        
        # --------------------------------------------------------
        # 3. Tính Loss Mục tiêu Chebyshev 
        # --------------------------------------------------------
        diff = y_pred_batch - z_star_tensor
        weighted_diff = r_tensor_batch * diff
        max_vals, _ = torch.max(weighted_diff, dim=1)
        L_Obj = torch.mean(max_vals)
        
        # --------------------------------------------------------
        # 4. Tổng hợp và Cập nhật 
        # --------------------------------------------------------
        # L_Total = L_Obj + β_C * L_C + β_Q * L_Q
        total_loss = L_Obj + (beta_C * L_bar_C) + (beta_Q * L_bar_Q)
        
        total_loss.backward()
        optimizer.step()
        scheduler.step()
        
        # --------------------------------------------------------
        # 5. Cập nhật hệ số Phạt 
        # --------------------------------------------------------
        # Cả hai hệ số đều TĂNG dần
        beta_C = min(beta_C_max, beta_C * rho_C)
        beta_Q = min(beta_Q_max, beta_Q * rho_Q)
        
        # Logging
        if verbose and epoch % 100 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch {epoch}: Total={total_loss.item():.3f} "
                  f"(Obj={L_Obj.item():.4f}, C={L_bar_C.item():.5f}, Q={L_bar_Q.item():.5f}) "
                  f"|| BetaC={beta_C:.1f}, BetaQ={beta_Q:.1f}")
            
    return hypernet