import itertools
from tqdm import tqdm
from Phase2_dynamic import optim_Scalarization
from evaluate import get_metrics
import pandas as pd

def generate_valid_params():
    grid = {
        'expo_lambda': [0.51, 0.65, 0.8],
        'expo_alpha':  [0.1, 0.25, 0.4], 
        'expo_beta':   [0.3, 0.5],
        'expo_gamma':  [0.3, 0.5],
        'init_params': [1.0, 2.0],
        'mu': [0.0001, 0.01],
        'max_iter' : [1000]
    }
    keys, values = zip(*grid.items())
    raw_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    valid_combinations = []
    for p in raw_combinations:
        # --- KIỂM TRA ĐIỀU KIỆN ---
        cond1 = 0.5 < p['expo_lambda'] <= 1.0
        cond2 = p['expo_lambda'] + p['expo_alpha'] <= 1
        cond3 = p['expo_beta'] < p['expo_lambda']
        cond4 = p['expo_gamma'] < p['expo_lambda']
        if cond1 and cond2 and cond3 and cond4:
            valid_combinations.append(p)  
    return valid_combinations

def tuning_dynamic(prob, z_star, x_feasible, pf_true, ref_point, test_rays):
    param_combinations = generate_valid_params()
    results_log = []

    print(f"=== GRID SEARCH: ƯU TIÊN MED ===")
    print(f"Số lượng cấu hình: {len(param_combinations)} | Số tia: {len(test_rays)}")

    for idx, params in tqdm(enumerate(param_combinations)):
        pareto_f_temp = []
        for r in test_rays:
            x_final, _ = optim_Scalarization(
                prob=prob,
                x_feasible=x_feasible,  
                r=r, 
                z_star=z_star,
                verbose=False,
                **params 
            )
            pareto_f_temp.append(prob.objective_func(x_final))
        med, hv = get_metrics(pareto_f_temp, pf_true, prob, ref_point)

        log_entry = {
            'params': params,
            'MED': med,
            'HV': hv,
            'id': idx
        }
        results_log.append(log_entry)

        print(f"[{idx:02d}] MED: {med:.6e} | HV: {hv:.6f}")
    best_entry = min(results_log, key=lambda x: x['MED'])
    print("\n" + "="*60)
    print(f"CẤU HÌNH TỐI ƯU NHẤT THEO MED:")
    print(f"ID: {best_entry['id']}")
    print(f"MED: {best_entry['MED']:.6f}, HV {best_entry['HV']}")
    print(f"Params: {best_entry['params']}")
    print("="*60)
    df = pd.json_normalize(results_log, sep='_')
    df.columns = df.columns.str.replace('params_', '')
    
    return best_entry['params'], df