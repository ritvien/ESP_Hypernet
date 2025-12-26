 
import yaml
import autograd.numpy as np
import pandas as pd
from Phase1 import solve_CQ_feasible
from Phase2_dynamic import optim_Scalarization
from Phase2_1_obj import optim_Universal
from Problem import Problem

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    cfg['initialization']['x_init'] = np.array(cfg['initialization']['x_init'])
    cfg['data']['test_ray'] = np.array(cfg['data']['test_ray'])
    return cfg

def run_experiment(prob):
    cfg = load_config()
        
    print("=== BẮT ĐẦU PHASE 1: TÌM ĐIỂM KHẢ THI ===")
    x_feasible, x_hist_p1, f_hist_p1, z_proj_hist = solve_CQ_feasible(
        f=prob.objective_func,
        jac_f=prob.jacobian,
        proj_C=prob.proj_C,
        proj_Qplus=prob.proj_Qplus,
        x0=cfg['initialization']['x_init'],
        gamma=cfg['phase1']['gamma'], 
        expo_gamma=cfg['phase1']['expo_gamma'],
        max_iter=cfg['phase1']['max_iter'],
        tol=cfg['phase1']['tol']
    )
    print(f"-> Điểm khả thi (Feasible Point): {x_feasible}")
    
    print("=== TÌM GIỚI HẠN PARETO: OPTIM 1 OBJ CỦA F ===")
    limit_Q = []
    for dim in range(2):
        x_final, _ = optim_Universal(
                prob=prob,
                x_feasible=x_feasible,  
                r=None,
                target_dim=dim,
                mode="min",
                max_iter=cfg['phase2']['max_iter'],
                mu=cfg['phase2']['mu'],
                expo_alpha=cfg['phase2']['expo_alpha'],
                expo_lambda=cfg['phase2']['expo_lambda'],
                init_params=cfg['phase2']['init_params'],

            )
        limit_Q.append(prob.objective_func(x_final)[dim])
        print(f"Chiều {dim}: {prob.objective_func(x_final)[dim]}")
    z_star = np.array(limit_Q)
#     z_star = np.array([0.0, 0.0])

    print("\n=== BẮT ĐẦU PHASE 2: SCALARIZATION (MULTI-RAY) ===")
    pareto_front_x = [] 
    pareto_front_f = [] 
    all_paths = []      
    
    test_rays = cfg['data']['test_ray']
    
    for i, r in enumerate(test_rays):
        # In tiến độ
        print(f"Running Ray {i+1}/{len(test_rays)}: {r}")
        
        x_final, path_x = optim_Scalarization(
            prob=prob,
            x_feasible=x_feasible,  
            r=r, 
            z_star=z_star,
            max_iter=cfg['phase2']['max_iter'],
            mu=cfg['phase2']['mu'],
            expo_alpha=cfg['phase2']['expo_alpha'],
            expo_lambda=cfg['phase2']['expo_lambda'],
            init_params=cfg['phase2']['init_params'],
            expo_beta=cfg['phase2']['expo_beta'],
            verbose=cfg['phase2']['verbose']
        )
        
        pareto_front_x.append(x_final)
        pareto_front_f.append(prob.objective_func(x_final)) 
        all_paths.append(path_x)

    pareto_front_f = np.array(pareto_front_f)
    pareto_front_x = np.array(pareto_front_x)
    
    print("\n=== HOÀN THÀNH ===")
    print(f"Tìm thấy {len(pareto_front_x)} điểm Pareto.")

    print(pareto_front_f)
    return {
        "x_feasible_phase1": x_feasible,
        "pareto_x": np.array(pareto_front_x),
        "pareto_f": np.array(pareto_front_f),
        "all_paths": all_paths
    }
