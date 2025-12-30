from functions_hv_python3 import HyperVolume
import autograd.numpy as np
import torch


def get_metrics(pf_pred, pf_true, prob, ref_point):
    """Tính toán đồng thời MED và HV"""
    pf_pred = np.array(pf_pred)
    
    # --- 1. Tính MED (Mean Error Distance) ---
    # Vì dùng chung tập test_ray nên ta tính khoảng cách cặp 1-1
    med = np.mean(np.linalg.norm(pf_pred - pf_true, axis=1))
    
    # --- 2. Tính HV (HyperVolume) ---
    valid_points = []
    tol = 1e-3
    for point in pf_pred:
        point_proj = prob.proj_Qplus(point)
        dist_Q = np.linalg.norm(point - point_proj)
        is_dominated_by_ref = np.all(point < ref_point)
        
        if dist_Q < tol and is_dominated_by_ref:
            valid_points.append(point.tolist())
            
    hv = 0.0
    if len(valid_points) >= 2:
        hv_obj = HyperVolume(ref_point)
        hv = hv_obj.compute(valid_points)
        
    return med, hv

def evaluate_model_all(hypernet, prob, test_rays, pf_true, ref_point, device):
    """
    Đánh giá model một lần duy nhất cho cả MED và HV.
    """
    hypernet.eval()
    pf_pred = []
    rays_tensor = torch.tensor(test_rays, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        for i in range(len(rays_tensor)):
            r_single = rays_tensor[i].unsqueeze(0)
            x_raw = hypernet(r_single).squeeze().cpu().numpy()
            
            # Chiếu lên C để đảm bảo x hợp lệ
            x_proj = prob.proj_C(x_raw)
            
            # Tính f(x)
            val = [func(x_proj) for func in prob.f]
            pf_pred.append(val)
            
    pf_pred = np.array(pf_pred)
    
    # Tính các chỉ số
    med_score, hv_score = get_metrics(pf_pred, pf_true, prob, ref_point)
    
    return med_score, hv_score, pf_pred