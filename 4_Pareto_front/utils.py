import matplotlib.pyplot as plt
import numpy as np

def visualize_pareto_front(results, pf_cloud=None, pf_targets=None, title="Pareto Front Comparison", save_path=None, figsize=(10, 8)):
    """
    Vẽ không gian mục tiêu (Objective Space) so sánh giữa thực tế và dự đoán.
    
    Args:
        results (dict): Kết quả trả về từ hàm run_experiment (chứa 'pareto_f').
        pf_cloud (np.array): Tập hợp lớn các điểm Pareto tham chiếu (màu xám).
        pf_targets (np.array): Các điểm Pareto đích thực sự tương ứng với tia test (màu vàng).
        title (str): Tiêu đề biểu đồ.
        save_path (str): Đường dẫn lưu file ảnh.
        figsize (tuple): Kích thước ảnh.
    """
    
    # Lấy dữ liệu dự đoán từ kết quả chạy
    obs_predict = results.get('pareto_f')
    
    # Tạo plot với kích thước chuẩn
    fig, ax = plt.subplots(figsize=figsize)
    
    # 1. Vẽ đám mây điểm Pareto tham chiếu (Màu xám - Nền)
    if pf_cloud is not None:
        ax.scatter(
            pf_cloud[:, 0], pf_cloud[:, 1], 
            s=20, c="gray", marker="X", alpha=0.6, 
            label="PF (Reference Cloud)"
        )

    # 2. Vẽ các điểm đích thực sự (Màu vàng - Target)
    if pf_targets is not None:
        ax.scatter(
            pf_targets[:, 0], pf_targets[:, 1], 
            s=50, c="yellow", marker="X", 
            label="PF_true (Targets)", zorder=5
        )

    # 3. Vẽ các điểm thuật toán tìm được (Màu đỏ - Predict)
    if obs_predict is not None and len(obs_predict) > 0:
        ax.scatter(
            obs_predict[:, 0], obs_predict[:, 1], 
            s=60, c="red", marker="D", 
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