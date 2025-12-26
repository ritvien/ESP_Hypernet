import matplotlib.pyplot as plt
import numpy as np

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