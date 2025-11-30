import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.optimize import Bounds

def select_interesting_pairs(data, max_pairs=3):
    """
    Chọn ra các cặp chiều (indices) đáng quan tâm nhất dựa trên sự thay đổi (Standard Deviation).
    """
    n_dim = data.shape[1]
    if n_dim < 2: return []
    
    if n_dim == 2: return [(0, 1)]
    if n_dim == 3: return [(0, 1), (0, 2), (1, 2)]
    
    stds = np.std(data, axis=0)
    top_dims = np.argsort(stds)[::-1] 
    
    pairs = []
    for i in range(min(n_dim, max_pairs + 1)):
        for j in range(i + 1, min(n_dim, max_pairs + 1)):
            pairs.append((top_dims[i], top_dims[j]))
            if len(pairs) >= max_pairs: return pairs
    return pairs

def draw_subplot_slice(ax, d1, d2, data_main, data_proj, constraints, bounds, title_prefix, raw_point=None):
    """
    Hàm phụ trợ để vẽ 1 subplot đơn lẻ.
    Có hỗ trợ vẽ điểm raw_point (x0 hoặc z0) nằm ngoài quỹ đạo.
    """
    has_proj = data_proj is not None
    
    # 1. Xác định tập dữ liệu để tính giới hạn vẽ
    if has_proj:
        all_data = np.vstack((data_main, data_proj))
        fixed_point = data_proj[-1] 
    else:
        all_data = data_main
        fixed_point = data_main[-1]

    # Nếu có raw_point, cần thêm nó vào tính toán giới hạn để không bị scale mất
    if raw_point is not None:
        all_data = np.vstack((all_data, raw_point))

    min_x, max_x = np.min(all_data[:, d1]), np.max(all_data[:, d1])
    min_y, max_y = np.min(all_data[:, d2]), np.max(all_data[:, d2])
    
    pad_x = (max_x - min_x) * 1 + 0.5
    pad_y = (max_y - min_y) * 1 + 0.5
    
    xlims = (min_x - pad_x, max_x + pad_x)
    ylims = (min_y - pad_y, max_y + pad_y)
    
    # 2. Quét miền chấp nhận được (Feasible Region)
    resolution = 300 
    xx = np.linspace(xlims[0], xlims[1], resolution)
    yy = np.linspace(ylims[0], ylims[1], resolution)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = np.zeros_like(XX)
    
    for i in range(resolution):
        for j in range(resolution):
            sample_pt = fixed_point.copy()
            sample_pt[d1] = XX[i, j]
            sample_pt[d2] = YY[i, j]
            
            satisfy = True
            if bounds is not None:
                if isinstance(bounds, (list, tuple)): 
                     if bounds[d1] and ((bounds[d1][0] is not None and sample_pt[d1] < bounds[d1][0]) or (bounds[d1][1] is not None and sample_pt[d1] > bounds[d1][1])): satisfy = False
                     if bounds[d2] and ((bounds[d2][0] is not None and sample_pt[d2] < bounds[d2][0]) or (bounds[d2][1] is not None and sample_pt[d2] > bounds[d2][1])): satisfy = False

            if satisfy and constraints:
                for cons in constraints:
                    val = cons['fun'](sample_pt)
                    if cons['type'] == 'ineq': 
                        if np.any(val < -1e-5): satisfy = False; break
                    elif cons['type'] == 'eq':
                        if not np.allclose(val, 0, atol=1e-3): satisfy = False; break
            
            ZZ[i, j] = 1.0 if satisfy else 0.0

    # Vẽ Contour miền
    ax.contourf(XX, YY, ZZ, levels=[0.5, 1.5], colors=['#90EE90'], alpha=0.4)
    ax.contour(XX, YY, ZZ, levels=[0.5], colors='green', linewidths=1.0)
    
    # 3. Vẽ điểm Raw Input (x0 hoặc z0) nếu có
    if raw_point is not None:
        # Vẽ điểm
        ax.scatter(raw_point[d1], raw_point[d2], c='green', s=70, marker='s', label='Raw Input', zorder=15)
        # Vẽ đường nối đứt đoạn từ Raw Input đến điểm bắt đầu của quỹ đạo
        ax.plot([raw_point[d1], data_main[0, d1]], 
                [raw_point[d2], data_main[0, d2]], 
                color='black', linestyle=':', linewidth=1.5, alpha=0.8)

    # 4. Vẽ quỹ đạo chính
    if has_proj:
        # Chế độ so sánh (f(x) vs z)
        step = max(1, len(data_main) // 30)
        for k in range(0, len(data_main), step):
            ax.plot([data_main[k, d1], data_proj[k, d1]], 
                    [data_main[k, d2], data_proj[k, d2]], 
                    color='gray', alpha=0.3, linewidth=0.5)
        
        ax.plot(data_main[:, d1], data_main[:, d2], 'r-o', markersize=2, linewidth=1, label='f(x)')
        ax.plot(data_proj[:, d1], data_proj[:, d2], 'b--.', markersize=1, linewidth=0.8, alpha=0.7, label='Proj')
        ax.scatter(data_main[-1, d1], data_main[-1, d2], c='purple', s=60, marker='*', zorder=10)
    else:
        # Chế độ đơn (x)
        ax.plot(data_main[:, d1], data_main[:, d2], 'r-o', markersize=3, linewidth=1.5, label='x')
        # Điểm đầu của quỹ đạo (Sau khi đã chiếu x0 lên C)
        ax.scatter(data_main[0, d1], data_main[0, d2], c='black', s=40, marker='x', label='Start (Proj)')
        ax.scatter(data_main[-1, d1], data_main[-1, d2], c='blue', s=60, marker='*')

    ax.set_title(f"{title_prefix}: Dim {d1} vs {d2}", fontsize=10)
    ax.set_xlabel(f"D{d1}", fontsize=9)
    ax.set_ylabel(f"D{d2}", fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3)

def visualize_complete_system(x_hist, f_hist, z_proj_hist, 
                              cons_C, cons_Q, 
                              bounds_x=None, bounds_y=None,
                              max_cols=3,
                              x0=None, z0=None): # Thêm tham số x0, z0
    """
    Hàm vẽ All-in-One:
    - Hàng 1: Các cặp chiều của X (Miền C). Có thể hiển thị x0.
    - Hàng 2: Các cặp chiều của f(x) vs z_proj (Miền Q). Có thể hiển thị z0.
    """
    print("Đang xử lý dữ liệu và tạo đồ thị tổng hợp...")
    
    data_x = np.array(x_hist)
    data_f = np.array(f_hist)
    data_z = np.array(z_proj_hist)
    
    n = min(len(data_x), len(data_f), len(data_z))
    data_x = data_x[:n]
    data_f = data_f[:n]
    data_z = data_z[:n]
    
    pairs_x = select_interesting_pairs(data_x, max_pairs=max_cols)
    pairs_y = select_interesting_pairs(data_f, max_pairs=max_cols)
    
    n_cols = max(len(pairs_x), len(pairs_y))
    
    fig, axes = plt.subplots(nrows=2, ncols=n_cols, figsize=(4 * n_cols, 8))
    if n_cols == 1: axes = np.array([[axes[0]], [axes[1]]])
    
    # --- VẼ HÀNG 1: MIỀN C (BIẾN X) ---
    for i in range(n_cols):
        ax = axes[0, i]
        if i < len(pairs_x):
            d1, d2 = pairs_x[i]
            # Truyền x0 vào raw_point
            draw_subplot_slice(ax, d1, d2, data_x, None, cons_C, bounds_x, "Domain C", raw_point=x0)
        else:
            ax.axis('off')

    # --- VẼ HÀNG 2: MIỀN Q (BIẾN f(x)) ---
    for i in range(n_cols):
        ax = axes[1, i]
        if i < len(pairs_y):
            d1, d2 = pairs_y[i]
            # Truyền z0 vào raw_point (nếu có)
            draw_subplot_slice(ax, d1, d2, data_f, data_z, cons_Q, bounds_y, "Image Q", raw_point=z0)
        else:
            ax.axis('off')

    # Legend chung
    legend_elements = [
        Patch(facecolor='#90EE90', edgecolor='green', alpha=0.4, label='Feasible Region'),
        Line2D([0], [0], color='green', marker='s', linestyle='None', markersize=8, label='Raw Input (x0/z0)'),
        Line2D([0], [0], color='red', marker='o', markersize=5, label='Trajectory'),
        Line2D([0], [0], color='blue', linestyle='--', label='Projection'),
        Line2D([0], [0], color='purple', marker='*', linestyle='None', markersize=10, label='Converged Point')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1) 
    
    return fig