import os
import random
import numpy as np
import matplotlib.pyplot as plt

# 1. Cấu hình các mốc kích thước (Đúng 10 mốc từ nhỏ đến lớn)
# Điều này giúp bạn sinh ra đúng 10 test/phân phối để test khả năng mở rộng thuật toán
test_sizes = [20, 50, 80, 100, 150, 200, 300, 500, 800, 1000]
distributions = ["Uniform", "Gaussian", "Poisson", "Exponential", "Adversarial"]

output_dir = "Datasets"
os.makedirs(output_dir, exist_ok=True)

# Nơi lưu trữ data để vẽ biểu đồ (chỉ lấy mẫu từ mốc 1000 để biểu đồ rõ nét nhất)
plot_data = {dist: {"teachers_per_subject": [], "subjects_per_teacher": []} for dist in distributions}

def get_num_teachers_for_subject(dist, T):
    """Giới hạn thực tế: 1 môn tối đa khoảng 10% tổng số giáo viên có thể dạy"""
    max_teachers = max(3, int(T * 0.1)) 
    
    if dist == "Uniform":
        return random.randint(1, max_teachers)
    elif dist == "Gaussian":
        mu, sigma = max_teachers / 2, max_teachers / 6
        val = int(round(random.gauss(mu, sigma)))
        return max(1, min(val, max_teachers))
    elif dist == "Poisson":
        val = np.random.poisson(lam=max_teachers / 3)
        return max(1, min(val, max_teachers))
    elif dist == "Exponential":
        val = int(round(np.random.exponential(scale=max_teachers / 4)))
        return max(1, min(val, max_teachers))
    elif dist == "Adversarial":
        # Adversarial (Bait & Trap): 20% môn hiếm (1-2 người dạy), 80% môn đại trà
        if random.random() < 0.2:
            return random.randint(1, 2)
        else:
            return random.randint(max_teachers // 2, max_teachers)
    return 1

def generate_and_save_test(dist_name, base_size):
    # Thêm nhiễu ngẫu nhiên (+- 10%) để T, M, N không bằng nhau chằn chặn
    T = max(10, int(base_size * random.uniform(0.9, 1.1)))
    N = max(10, int(base_size * random.uniform(0.9, 1.1)))
    M = max(10, int(base_size * random.uniform(0.7, 0.9))) # Số môn thường ít hơn số Lớp/Giáo viên
    
    # Sinh d(m): Số tiết của mỗi môn (từ 2 đến 4 tiết)
    d = {m: random.randint(2, 4) for m in range(1, M + 1)}
    
    # Cấu trúc lưu trữ
    teacher_subjects = {t: set() for t in range(1, T + 1)}
    subject_teachers_count = [] 
    
    # BƯỚC 1: Phân công Giáo viên cho Môn học
    for m in range(1, M + 1):
        num_teachers = get_num_teachers_for_subject(dist_name, T)
        subject_teachers_count.append(num_teachers)
        
        assigned_teachers = random.sample(range(1, T + 1), num_teachers)
        for t in assigned_teachers:
            teacher_subjects[t].add(m)
            
    # Lấy mẫu data để vẽ biểu đồ khi base_size = 1000
    if base_size == 1000:
        plot_data[dist_name]["teachers_per_subject"].extend(subject_teachers_count)
        plot_data[dist_name]["subjects_per_teacher"].extend([len(subs) for subs in teacher_subjects.values()])

    # BƯỚC 2: Phân công Môn học cho Lớp (Đảm bảo <= 55 tiết/tuần)
    class_subjects = {n: set() for n in range(1, N + 1)}
    for n in range(1, N + 1):
        current_slots = 0
        available_subjects = list(range(1, M + 1))
        random.shuffle(available_subjects)
        
        for m in available_subjects:
            if current_slots + d[m] <= 55:
                class_subjects[n].add(m)
                current_slots += d[m]
            else:
                break
                
        # Nếu lớp xui xẻo không bốc được môn nào, ép học ít nhất 1 môn
        if len(class_subjects[n]) == 0:
            class_subjects[n].add(random.randint(1, M))

    # BƯỚC 3: Ghi ra file text đúng định dạng đề bài
    filename = os.path.join(output_dir, f"{dist_name}_{T}_{M}_{N}.txt")
    with open(filename, 'w') as f:
        # Dòng 1: T, N, M
        f.write(f"{T} {N} {M}\n")
        
        # N dòng tiếp theo: Danh sách môn của lớp, kết thúc bởi 0
        for n in range(1, N + 1):
            subs = list(class_subjects[n])
            f.write(" ".join(map(str, subs)) + " 0\n")
            
        # T dòng tiếp theo: Danh sách môn của giáo viên, kết thúc bởi 0
        for t in range(1, T + 1):
            subs = list(teacher_subjects[t])
            if not subs:
                f.write("0\n")
            else:
                f.write(" ".join(map(str, subs)) + " 0\n")
                
        # Dòng cuối: d(m)
        f.write(" ".join(str(d[m]) for m in range(1, M + 1)) + "\n")

# --- THỰC THI CHÍNH ---
print("Bắt đầu sinh bộ dữ liệu Test...")
for dist in distributions:
    for size in test_sizes:
        generate_and_save_test(dist, size)
    print(f" Đã sinh xong 10 test cho phân phối: {dist}")

# --- VẼ BIỂU ĐỒ (VISUALIZATION) ---
print("\nĐang phân tích và vẽ biểu đồ phân phối...")
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
fig.suptitle(f"Teacher-Subject Assignments Distributions (Sampling at Size=1000)", fontsize=18, fontweight='bold')

for i, dist in enumerate(distributions):
    # Hàng 1: Số Giáo viên / Môn học (Giống ks trong bài Reviewer)
    axes[0, i].hist(plot_data[dist]["teachers_per_subject"], bins=20, color='#2874A6', edgecolor='black', alpha=0.85)
    axes[0, i].set_title(f"{dist}\nTeachers per Subject", fontsize=12)
    axes[0, i].set_xlabel("Number of Teachers")
    axes[0, i].set_ylabel("Frequency (Subjects)")
    axes[0, i].grid(axis='y', linestyle='--', alpha=0.7)
    
    # Hàng 2: Số Môn học / Giáo viên
    axes[1, i].hist(plot_data[dist]["subjects_per_teacher"], bins=20, color='#117A65', edgecolor='black', alpha=0.85)
    axes[1, i].set_title(f"Subjects per Teacher", fontsize=12)
    axes[1, i].set_xlabel("Number of Subjects")
    axes[1, i].set_ylabel("Frequency (Teachers)")
    axes[1, i].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
image_dir = os.path.join("Figure", "Data")

os.makedirs(image_dir, exist_ok=True)

image_path = os.path.join(image_dir, "Data_Distributions_Analysis.png")

plt.savefig(image_path, dpi=300, bbox_inches='tight')
print(f"Hoàn tất! Kiểm tra thư mục 'Datasets' và file ảnh '{image_path}'.")