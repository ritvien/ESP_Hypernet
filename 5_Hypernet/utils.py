import numpy as np
import matplotlib.pyplot as plt
from pymoo.util.reference_direction import UniformReferenceDirectionFactory


def get_ref_dirs(n_obj):
    if n_obj == 2:
        ref_dirs = UniformReferenceDirectionFactory(2, n_points=25).do()
    elif n_obj == 3:
        #ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=15).do()
        ref_dirs = UniformReferenceDirectionFactory(3, n_partitions=100).do()
    else:
        raise Exception("Please provide reference directions for more than 3 objectives!")
    return ref_dirs

def circle_points(K, min_angle=None, max_angle=None):
    # generate evenly distributed preference vector
    ang0 = 1e-6 if min_angle is None else min_angle
    ang1 = np.pi / 2 - ang0 if max_angle is None else max_angle
    angles = np.linspace(ang0, ang1, K, endpoint=True)
    x = np.cos(angles)
    y = np.sin(angles)
    return np.c_[x, y]

def plot_trajectories(obs_true=None, obs_predict=None, obs_pareto_front=None, title = None, save=None, figsize=(16, 8)):
    plt.figure(figsize=figsize)
    ig, ax = plt.subplots()
    ax.scatter(
        obs_pareto_front[:, 0], obs_pareto_front[:, 1], s=20, c="gray", label="PF", marker="X"
    )
    ax.scatter(
        obs_true[:, 0], obs_true[:, 1], s=20, c="yellow", label="PF_true", marker="D"
    )
    ax.scatter(
        obs_predict[:, 0],
        obs_predict[:, 1],
        s=20,
        c="red",
        label="PF_predict",
        marker="X",
    )
    plt.legend()
    ax.set_title(title)
    if save:
        plt.savefig(save_path)
    plt.show()
    
def plot_solution_space(target_space, target_space_full, target_space_cir):
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True, sharey=True)

    axes[0].scatter(target_space_full[:, 0], target_space_full[:, 1], s=10, c="red", label="X - Constraints")
    axes[0].legend()

    axes[0].scatter(target_space_cir[:, 0], target_space_cir[:, 1], s=10, c="yellow", label="Circle - (f - constraint)")
    axes[0].legend()

    axes[1].scatter(target_space[:, 0], target_space[:, 1], s=10, c="gray", label="Feasible solutions space")
    axes[1].legend()

    # Đảm bảo tỷ lệ trục bằng nhau
    for ax in axes:
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.show()
    
    import matplotlib.pyplot as plt
import numpy as np

def generate_pareto_grid(f_func, c_funcs, q_plus_func, resolution=400):
    """
    Sinh dữ liệu Pareto cho bài toán 2D bằng cách quét lưới tọa độ.
    f_func: Hàm mục tiêu f(x)
    c_funcs: Danh sách các hàm ràng buộc c(x) >= 0 (miền C)
    q_plus_func: Hàm ràng buộc q_plus(y) >= 0 (miền Q+)
    """
    print(f"--- Đang quét lưới {resolution}x{resolution} điểm ---")
    
    # 1. Xác định phạm vi quét dựa trên các ràng buộc hình tròn c1, c2

    x0_range = np.linspace(0, 1, resolution)
    x1_range = np.linspace(0, 1, resolution)
    X0, X1 = np.meshgrid(x0_range, x1_range)
    
    # Làm phẳng để tính toán vector
    points_X = np.vstack([X0.ravel(), X1.ravel()]).T
    
    # 2. Kiểm tra ràng buộc miền C: c1(x) >= 0 và c2(x) >= 0
    # (Sử dụng các hàm c1, c2 đã định nghĩa của bạn)
    mask_C = np.ones(len(points_X), dtype=bool)
    for c_func in c_funcs:
        mask_C &= (c_func(points_X.T) >= 0)
    
    feasible_X = points_X[mask_C]
    print(f"-> Tìm thấy {len(feasible_X)} điểm thỏa mãn miền C.")
    
    if len(feasible_X) == 0:
        return None, None

    # 3. Tính giá trị mục tiêu f(x) và kiểm tra miền Q+
    f_vals = np.array([f_func(x) for x in feasible_X])  # shape (N, 2)
    
    # Kiểm tra q_plus(f(x)) >= 0
    # Lưu ý: Hàm q_plus của bạn nhận 1 vector y, ta cần áp dụng cho toàn bộ f_vals
    mask_Qplus = np.array([q_plus_func(y) >= 0 for y in f_vals])
    
    pf_cloud_data = f_vals[mask_Qplus]
    print(f"-> Lọc còn {len(pf_cloud_data)} điểm thỏa mãn Q+ (Cloud).")

    if len(pf_cloud_data) == 0:
        return None, None

    # 4. Lọc Pareto Front từ Cloud (Tìm các điểm không bị trội)
    # Sắp xếp theo f1 tăng dần
    sorted_indices = np.argsort(pf_cloud_data[:, 0])
    y_sorted = pf_cloud_data[sorted_indices]
    
    pareto_list = []
    pareto_list.append(y_sorted[0])
    current_min_f2 = y_sorted[0][1]
    
    for i in range(1, len(y_sorted)):
        if y_sorted[i][1] < current_min_f2:
            pareto_list.append(y_sorted[i])
            current_min_f2 = y_sorted[i][1]
    
    pf_targets_data = np.array(pareto_list)
    print(f"-> Kết quả: {len(pf_targets_data)} điểm Pareto.")
    
    return pf_cloud_data, pf_targets_data

def visualize_pareto_front(pf_pred=None, pf_cloud=None, pf_targets=None, title="Pareto Front Comparison", save_path=None, figsize=(10, 8)):
    """
    Vẽ không gian mục tiêu (Objective Space) so sánh giữa thực tế và dự đoán.
    """

    
    # Tạo plot với kích thước chuẩn
    fig, ax = plt.subplots(figsize=figsize)
    
    # 1. Vẽ đám mây điểm Pareto tham chiếu (Màu xám - Nền)
    if pf_cloud is not None:
        ax.scatter(
            pf_cloud[:, 0], pf_cloud[:, 1], 
            s=100, c="gray", marker="X", alpha=0.6, 
            label="PF (Reference Cloud)"
        )

    # 2. Vẽ các điểm đích thực sự (Màu vàng - Target)
    if pf_targets is not None:
        ax.scatter(
            pf_targets[:, 0], pf_targets[:, 1], 
            s=60, c="yellow", marker="X", 
            label="PF_true (Targets)", zorder=5
        )

    # 3. Vẽ các điểm thuật toán tìm được (Màu đỏ - Predict)
    if pf_pred is not None and len(pf_pred) > 0:
        ax.scatter(
            pf_pred[:, 0], pf_pred[:, 1], 
            s=30, c="red", marker="D", 
            label="PF_predict (Found)", zorder=10
        )

    # Trang trí
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("$f_1(x)$")
    ax.set_ylabel("$f_2(x)$")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)

    # Lưu và hiển thị
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Đã lưu hình ảnh tại: {save_path}")
        
    plt.show()
    