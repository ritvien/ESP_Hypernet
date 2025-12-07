import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patches as patches
from scipy.optimize import Bounds
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# ====================================================
# 1. HÀM XỬ LÝ DỮ LIỆU & CHỌN CẶP CHIỀU
# ====================================================

def sanitize_data(data):
    """
    Chuyển đổi dữ liệu về dạng chuẩn 2D (N_samples, N_features).
    Xử lý các trường hợp bị dư chiều như (N, 1, 2) hoặc (N, 2, 1).
    """
    arr = np.array(data)
    # Nếu là mảng 1 chiều (N,), coi như là (N, 1)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    
    # Nếu là mảng 3 chiều (do tích lũy vector hàng/cột)
    if arr.ndim == 3:
        n, d1, d2 = arr.shape
        if d1 == 1: # Dạng (N, 1, D) -> Gom về (N, D)
            return arr.reshape(n, d2)
        elif d2 == 1: # Dạng (N, D, 1) -> Gom về (N, D)
            return arr.reshape(n, d1)
            
    return arr

def sanitize_point(point):
    """
    Chuyển đổi điểm đơn lẻ về dạng chuẩn 1D (D,).
    """
    if point is None: return None
    arr = np.array(point)
    return arr.flatten() # Ép về mảng 1 chiều phẳng

def select_interesting_pairs(data, max_pairs=3):
    """
    Chọn cặp chiều có biến động lớn nhất.
    """
    # Dữ liệu đã được sanitize thành (N, D)
    n_dim = data.shape[1]
    
    if n_dim < 2: 
        # Nếu chỉ có 1 chiều thì không vẽ được 2D slice
        return []
        
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

# ====================================================
# 2. HÀM VẼ CỐT LÕI (MAIN & INSET)
# ====================================================

def plot_data_on_ax(ax, d1, d2, data_main, data_proj, raw_point, title_prefix, is_inset=False):
    has_proj = data_proj is not None
    
    # 1. Vẽ điểm Raw Input (chỉ vẽ trên hình to)
    if raw_point is not None and not is_inset:
        ax.scatter(raw_point[d1], raw_point[d2], c='green', s=70, marker='s', zorder=15, label='_nolegend_')
        ax.plot([raw_point[d1], data_main[0, d1]], 
                [raw_point[d2], data_main[0, d2]], 
                color='black', linestyle=':', linewidth=1.0, alpha=0.6)

    # 2. Vẽ quỹ đạo
    if has_proj:
        step = max(1, len(data_main) // (20 if is_inset else 50))
        for k in range(0, len(data_main), step):
            ax.plot([data_main[k, d1], data_proj[k, d1]], 
                    [data_main[k, d2], data_proj[k, d2]], 
                    color='gray', alpha=0.3, linewidth=0.5)
        
        ms = 3 if is_inset else 2
        ax.plot(data_main[:, d1], data_main[:, d2], 'r-o', markersize=ms, linewidth=1.5 if is_inset else 1)
        ax.plot(data_proj[:, d1], data_proj[:, d2], 'b--.', markersize=ms-1, linewidth=1, alpha=0.7)
        ax.scatter(data_main[-1, d1], data_main[-1, d2], c='purple', s=80 if is_inset else 60, marker='*', zorder=10)
    else:
        ms = 4 if is_inset else 3
        ax.plot(data_main[:, d1], data_main[:, d2], 'r-o', markersize=ms, linewidth=1.5)
        if not is_inset:
            ax.scatter(data_main[0, d1], data_main[0, d2], c='black', s=40, marker='x')
        ax.scatter(data_main[-1, d1], data_main[-1, d2], c='blue', s=80 if is_inset else 60, marker='*')
    
    if not is_inset:
        ax.set_title(f"{title_prefix}: Dim {d1} vs {d2}", fontsize=10)
        ax.set_xlabel(f"D{d1}", fontsize=9)
        ax.set_ylabel(f"D{d2}", fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3)
    else:
        ax.grid(True, linestyle=':', alpha=0.5)

def draw_subplot_slice(ax, d1, d2, data_main, data_proj, constraints, bounds, title_prefix, 
                       raw_point=None, enable_zoom=False):
    has_proj = data_proj is not None
    if has_proj:
        traj_data = np.vstack((data_main, data_proj))
        fixed_point = data_proj[-1]
    else:
        traj_data = data_main
        fixed_point = data_main[-1]

    # Tính giới hạn toàn cục
    global_data = traj_data
    if raw_point is not None:
        global_data = np.vstack((global_data, raw_point))
        
    g_min_x, g_max_x = np.min(global_data[:, d1]), np.max(global_data[:, d1])
    g_min_y, g_max_y = np.min(global_data[:, d2]), np.max(global_data[:, d2])
    
    pad = 0.2
    g_pad_x = (g_max_x - g_min_x) * pad + 0.5
    g_pad_y = (g_max_y - g_min_y) * pad + 0.5
    xlims_g = (g_min_x - g_pad_x, g_max_x + g_pad_x)
    ylims_g = (g_min_y - g_pad_y, g_max_y + g_pad_y)

    # --- VẼ NỀN (Feasible Region) ---
    resolution = 300
    xx = np.linspace(xlims_g[0], xlims_g[1], resolution)
    yy = np.linspace(ylims_g[0], ylims_g[1], resolution)
    XX, YY = np.meshgrid(xx, yy)
    ZZ = np.zeros_like(XX)
    
    for i in range(resolution):
        for j in range(resolution):
            sample_pt = fixed_point.copy()
            sample_pt[d1] = XX[i, j]
            sample_pt[d2] = YY[i, j]
            satisfy = True
            if bounds: 
                if isinstance(bounds, (list, tuple)):
                     if bounds[d1] and ((bounds[d1][0] is not None and sample_pt[d1] < bounds[d1][0]) or (bounds[d1][1] is not None and sample_pt[d1] > bounds[d1][1])): satisfy = False
                     if bounds[d2] and ((bounds[d2][0] is not None and sample_pt[d2] < bounds[d2][0]) or (bounds[d2][1] is not None and sample_pt[d2] > bounds[d2][1])): satisfy = False
            
            if satisfy and constraints:
                for cons in constraints:
                    val = cons['fun'](sample_pt)
                    if cons['type'] == 'ineq' and np.any(val < -1e-5): satisfy = False; break
                    elif cons['type'] == 'eq' and not np.allclose(val, 0, atol=1e-3): satisfy = False; break
            ZZ[i, j] = 1.0 if satisfy else 0.0

    ax.contourf(XX, YY, ZZ, levels=[0.5, 1.5], colors=['#90EE90'], alpha=0.4)
    ax.contour(XX, YY, ZZ, levels=[0.5], colors='green', linewidths=1.0)

    # --- VẼ HÌNH CHÍNH ---
    plot_data_on_ax(ax, d1, d2, data_main, data_proj, raw_point, title_prefix, is_inset=False)
    
    # --- LOGIC ZOOM (INSET) ---
    if enable_zoom:
        t_min_x, t_max_x = np.min(traj_data[:, d1]), np.max(traj_data[:, d1])
        t_min_y, t_max_y = np.min(traj_data[:, d2]), np.max(traj_data[:, d2])
        
        traj_span_x = t_max_x - t_min_x
        traj_span_y = t_max_y - t_min_y
        global_span_x = g_max_x - g_min_x
        global_span_y = g_max_y - g_min_y
        
        # Chỉ zoom nếu quỹ đạo nhỏ hơn 30% so với toàn cảnh
        if (traj_span_x < global_span_x * 0.3) or (traj_span_y < global_span_y * 0.3):
            axins = ax.inset_axes([0.5, 0.5, 0.48, 0.48]) 
            axins.contourf(XX, YY, ZZ, levels=[0.5, 1.5], colors=['#90EE90'], alpha=0.4)
            axins.contour(XX, YY, ZZ, levels=[0.5], colors='green', linewidths=1.0)
            plot_data_on_ax(axins, d1, d2, data_main, data_proj, None, "", is_inset=True)
            
            # Padding siêu nhỏ cho hình zoom (2%)
            zoom_pad_x = max(traj_span_x * 0.02, 0.01) 
            zoom_pad_y = max(traj_span_y * 0.02, 0.01)
            
            axins.set_xlim(t_min_x - zoom_pad_x, t_max_x + zoom_pad_x)
            axins.set_ylim(t_min_y - zoom_pad_y, t_max_y + zoom_pad_y)
            axins.set_xticklabels([])
            axins.set_yticklabels([])
            
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

# ====================================================
# 3. HÀM CHÍNH (ALL-IN-ONE)
# ====================================================

def visualize_complete_system(x_hist, f_hist, z_proj_hist, 
                              cons_C, cons_Q, 
                              bounds_x=None, bounds_y=None,
                              max_cols=3,
                              x0=None, z0=None):
    print("Đang xử lý dữ liệu và tạo đồ thị tổng hợp...")
    
    # 1. Sanitize Data (Quan trọng: Sửa lỗi dimension)
    data_x = sanitize_data(x_hist)
    data_f = sanitize_data(f_hist)
    data_z = sanitize_data(z_proj_hist)
    
    # 1b. Sanitize Points
    pt_x0 = sanitize_point(x0)
    pt_z0 = sanitize_point(z0)

    # 2. Đồng bộ độ dài
    n = min(len(data_x), len(data_f), len(data_z))
    data_x = data_x[:n]
    data_f = data_f[:n]
    data_z = data_z[:n]
    
    # 3. Chọn cặp
    pairs_x = select_interesting_pairs(data_x, max_pairs=max_cols)
    pairs_y = select_interesting_pairs(data_f, max_pairs=max_cols)
    
    n_cols = max(len(pairs_x), len(pairs_y))
    # Nếu không có cặp nào (ví dụ dữ liệu 1 chiều), n_cols = 0 -> không vẽ
    if n_cols == 0:
        print("Cảnh báo: Dữ liệu có số chiều < 2, không thể vẽ 2D Slice.")
        return None

    fig, axes = plt.subplots(nrows=2, ncols=n_cols, figsize=(4.5 * n_cols, 8.5))
    if n_cols == 1: axes = np.array([[axes[0]], [axes[1]]])
    
    # --- VẼ HÀNG 1: MIỀN C ---
    for i in range(n_cols):
        ax = axes[0, i]
        if i < len(pairs_x):
            d1, d2 = pairs_x[i]
            draw_subplot_slice(ax, d1, d2, data_x, None, cons_C, bounds_x, "Domain C", 
                               raw_point=pt_x0, enable_zoom=False)
        else:
            ax.axis('off')

    # --- VẼ HÀNG 2: MIỀN Q ---
    for i in range(n_cols):
        ax = axes[1, i]
        if i < len(pairs_y):
            d1, d2 = pairs_y[i]
            # Nếu dim_y = 2, len(pairs_y) = 1. Vẽ vào ô đầu tiên (i=0)
            draw_subplot_slice(ax, d1, d2, data_f, data_z, cons_Q, bounds_y, "Image Q", 
                               raw_point=pt_z0, enable_zoom=True)
        else:
            # Tắt ô thừa nếu X có 3 cặp mà Y chỉ có 1 cặp
            ax.axis('off')

    legend_elements = [
        Patch(facecolor='#90EE90', edgecolor='green', alpha=0.4, label='Feasible Region'),
        Line2D([0], [0], color='green', marker='s', linestyle='None', markersize=8, label='Raw Input'),
        Line2D([0], [0], color='red', marker='o', markersize=5, label='Trajectory'),
        Line2D([0], [0], color='blue', linestyle='--', label='Projection'),
        Line2D([0], [0], color='gray', linestyle='-', label='Zoom Region')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1) 
    
    return fig


def visualize_trajectory(path_x, prob, u_star, 
                         radius_c=2.0, 
                         q_bound=-1.0, 
                         x_limits=(-3.5, 2.5), 
                         y_limits=(-3.5, 1.5),
                         titles=None):
    """
    Hàm vẽ quỹ đạo tối ưu hóa trong không gian nguồn (C) và không gian ảnh (Q+).
    
    Tham số:
    - path_x: List hoặc array chứa lịch sử các điểm x qua từng bước lặp.
    - prob: Đối tượng bài toán (chứa phương thức objective_func).
    - u_star: Điểm mục tiêu mong muốn (trong không gian ảnh).
    - radius_c: Bán kính của tập ràng buộc C (mặc định 2.0).
    - q_bound: Giới hạn trên của tập Q+ (mặc định -1.0 cho cả 2 chiều).
    - x_limits: Tuple (min, max) cho trục của đồ thị X.
    - y_limits: Tuple (min, max) cho trục của đồ thị Y.
    - titles: Tuple hoặc List chứa 2 chuỗi ký tự (Title_X, Title_Y). 
              Ví dụ: ("Không gian biến", "Không gian ảnh").
              Nếu None, sẽ dùng title mặc định.
    """
    
    # 1. Chuẩn bị dữ liệu quỹ đạo
    path_x_arr = np.array(path_x) 
    
    # Tính toán quỹ đạo f(x) tương ứng
    path_f_arr = np.array([prob.objective_func(p) for p in path_x_arr])

    # 2. TÍNH TOÁN BIÊN CỦA f(C)
    # Tạo các điểm trên biên của C (Hình tròn) để ánh xạ sang bên phải
    theta = np.linspace(0, 2*np.pi, 200) # 200 điểm cho mượt
    boundary_x = radius_c * np.cos(theta)
    boundary_y = radius_c * np.sin(theta)
    boundary_C_points = np.vstack((boundary_x, boundary_y)).T

    # Ánh xạ các điểm biên này qua hàm f để có biên của f(C)
    boundary_fC_points = np.array([prob.objective_func(p) for p in boundary_C_points])

    # 3. Khởi tạo khung hình
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # --- XỬ LÝ TIÊU ĐỀ (TITLES) ---
    title_x = "Không gian biến $X$" # Mặc định
    title_y = "Không gian ảnh $Y$"  # Mặc định
    
    if titles is not None:
        if len(titles) >= 1: title_x = titles[0]
        if len(titles) >= 2: title_y = titles[1]

    # =========================================================
    # HÌNH 1: KHÔNG GIAN BIẾN X (Tập C và quỹ đạo x)
    # =========================================================
    ax1.set_title(title_x, fontsize=14) # <--- SỬ DỤNG TITLE X

    # Vẽ Tập C (Hình tròn)
    circle = plt.Circle((0, 0), radius_c, color='skyblue', alpha=0.3, label='Miền C')
    ax1.add_patch(circle)
    ax1.add_patch(plt.Circle((0, 0), radius_c, color='blue', fill=False, linestyle='--')) # Viền

    # Vẽ quỹ đạo x
    ax1.plot(path_x_arr[:, 0], path_x_arr[:, 1], 'k.-', linewidth=1.5, label='Quỹ đạo $x$')
    ax1.scatter(path_x_arr[0, 0], path_x_arr[0, 1], c='cyan', s=100, edgecolors='k', label='Start')
    ax1.scatter(path_x_arr[-1, 0], path_x_arr[-1, 1], c='red', s=120, marker='X', edgecolors='k', label='Final $x$')

    # Trang trí Ax1
    ax1.set_xlim(x_limits[0], x_limits[1])
    ax1.set_ylim(x_limits[0], x_limits[1]) 
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.set_xlabel("$x_1$")
    ax1.set_ylabel("$x_2$")
    ax1.legend(loc='upper right')

    # =========================================================
    # HÌNH 2: KHÔNG GIAN ẢNH Y (Ảnh f(C), Q+ và quỹ đạo f(x))
    # =========================================================
    ax2.set_title(title_y, fontsize=14) # <--- SỬ DỤNG TITLE Y

    # Vẽ Tập Ảnh f(C) dùng Polygon
    poly_fC = plt.Polygon(boundary_fC_points, closed=True, facecolor='mediumpurple', alpha=0.4, label='Ảnh $f(C)$')
    ax2.add_patch(poly_fC)
    # Vẽ đường viền của f(C)
    ax2.plot(boundary_fC_points[:, 0], boundary_fC_points[:, 1], color='indigo', linestyle='--', linewidth=1)

    # Vẽ biên giới hạn của Q+
    ax2.axvline(x=q_bound, color='red', linestyle='--', linewidth=1.5, label=f'Biên $Q^+$')
    ax2.axhline(y=q_bound, color='red', linestyle='--', linewidth=1.5)

    # Tô màu vùng Q+ (Vùng khả thi của f(x))
    rect_width = abs(y_limits[0] - q_bound) + 5 
    rect_height = abs(y_limits[0] - q_bound) + 5
    rect_Qplus = patches.Rectangle((q_bound - rect_width, q_bound - rect_height), 
                                   rect_width, rect_height, 
                                   color='salmon', alpha=0.1, label='Miền $Q^+$')
    ax2.add_patch(rect_Qplus)

    # Vẽ điểm mục tiêu u* (Target)
    ax2.scatter(u_star[0], u_star[1], c='gold', s=300, marker='*', edgecolors='k', zorder=10, label='Target $u^*$')

    # Vẽ quỹ đạo f(x)
    ax2.plot(path_f_arr[:, 0], path_f_arr[:, 1], 'k.-', linewidth=1.5, label='Quỹ đạo $f(x)$')
    ax2.scatter(path_f_arr[-1, 0], path_f_arr[-1, 1], c='red', s=120, marker='X', edgecolors='k', label='Final $f(x)$')

    # Trang trí Ax2
    ax2.set_xlim(y_limits[0], y_limits[1])
    ax2.set_ylim(y_limits[0], y_limits[1])
    ax2.set_aspect('equal')
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.set_xlabel("$f_1(x)$")
    ax2.set_ylabel("$f_2(x)$")
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.show()