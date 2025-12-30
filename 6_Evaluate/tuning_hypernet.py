import os
import time
import torch
import itertools
import yaml
import numpy as np
from tqdm import tqdm

from hypernet_MLP import Hypernet_MLP
from hypernet_trans import Hypernet_trans
from evaluate import evaluate_model_all

def run_hypernet_tuning(
    prob,
    dim_x,
    z_star,
    ref_point,
    test_rays,
    pf_true,
    param_grid,
    indicator="MED", 
    models_to_test=["MLP", "trans"],
    device="cpu",
    save_dir="models/tuning",
    train_func=None 
):
    """
    Hàm thực hiện tuning Hypernet cho cả 2 loại kiến trúc.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    results_log = []
    
    keys, values = zip(*param_grid.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for model_name in models_to_test:
        print(f"\n{"="*60}")
        print(f"🔹 TUNING MODEL: {model_name} | PRIORITY: {indicator}")
        print(f"{"="*60}")

        # Khởi tạo giá trị tốt nhất
        if indicator == "MED":
            current_best_score = float('inf')
        else:
            current_best_score = -1.0
            
        best_params_found = None

        for idx, params in enumerate(param_combinations):
            print(f"\n>>> [Config {idx+1}/{len(param_combinations)}] Params: {params}")
            
            # 1. Khởi tạo kiến trúc model tương ứng
            if model_name == "MLP":
                model = Hypernet_MLP(ray_hidden_dim=32, out_dim=dim_x, n_tasks=2).to(device)
            else:
                model = Hypernet_trans(ray_hidden_dim=32, out_dim=dim_x, n_tasks=2).to(device)

            # 2. Huấn luyện
            if train_func is None:
                raise ValueError("Bạn phải truyền hàm 'train_hypernet' vào tham số train_func.")
            
            start_time = time.time()
            trained_model = train_func(
                model, prob, z_star, 
                num_epochs=params['num_epochs'],
                lr=params['lr'],
                num_partitions=params['num_partitions'], 
                beta_C_0=params['beta_C_0'],
                beta_C_max=params['beta_C_max'],
                rho_C=params['rho_C'],
                beta_Q_0=params['beta_Q_0'],
                beta_Q_max=params['beta_Q_max'],
                rho_Q=params['rho_Q'],
                verbose=False 
            )
            train_time = time.time() - start_time

            # 3. Đánh giá 
            med, hv, _ = evaluate_model_all(trained_model, prob, test_rays, pf_true, ref_point, device)
            
            # 4. Xác định score dùng để so sánh
            score_to_compare = med if indicator == "MED" else hv
            
            print(f"      ⏱️ Time: {train_time:.1f}s | 📏 MED: {med:.6f} | 📈 HV: {hv:.6f}")

            # 5. Lưu kết quả vào log
            results_log.append({
                'model': model_name,
                'params': params,
                'med': med,
                'hv': hv,
                'time': train_time
            })

            # 6. Cập nhật và lưu model tốt nhất
            is_best = False
            if indicator == "MED":
                if score_to_compare < current_best_score:
                    current_best_score = score_to_compare
                    is_best = True
            else:
                if score_to_compare > current_best_score:
                    current_best_score = score_to_compare
                    is_best = True
            
            if is_best:
                best_params_found = params
                save_path = f"{save_dir}/best_{model_name}_{indicator}.pth"
                torch.save(trained_model.state_dict(), save_path)
                print(f"      🏆 NEW BEST FOUND! Saved to: {save_path}")

        print(f"\n✅ Hoàn thành {model_name}. Best {indicator}: {current_best_score:.6e}")
        print(f"   Best Config: {best_params_found}")

    return results_log