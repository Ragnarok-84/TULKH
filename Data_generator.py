import os
import random
import numpy as np
import matplotlib.pyplot as plt

# --- 1. CẤU HÌNH CÁC MỐC KÍCH THƯỚC ---
# test_sizes chính là Số lượng Lớp học (N). Ta dùng N làm mốc chuẩn.
test_sizes = [20, 50, 80, 100, 150, 200, 300, 500, 800, 1000]
distributions = ["Uniform", "Gaussian", "Poisson", "Exponential", "Adversarial"]

output_dir = "Datasets"
os.makedirs(output_dir, exist_ok=True)

plot_data = {dist: {"teachers_per_subject": [], "subjects_per_teacher": []} for dist in distributions}

def get_num_teachers_for_subject(dist, T):
    """Giới hạn: 1 môn chuyên ngành hẹp chỉ 1-2 người dạy, môn đại trà tối đa 10% giáo viên"""
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
        # Bẫy cục bộ: 20% môn siêu hiếm (chỉ 1 người dạy), 80% môn ai cũng dạy được
        if random.random() < 0.2:
            return 1
        else:
            return random.randint(max_teachers // 2, max_teachers)
    return 1

def generate_and_save_test(dist_name, base_size):
    # --- 2. ÁP DỤNG TỶ LỆ TRƯỜNG HỌC THỰC TẾ ---
    N = base_size                     # Cố định N (Lớp học) làm mốc chuẩn
    T = int(N * 1.5)                  # T (Giáo viên) = 150% số lớp
    M = max(10, int(N * 0.4))         # M (Môn học) = 40% số lớp
    
    # Sinh d(m): Số tiết của mỗi môn (từ 2 đến 4 tiết)
    d = {m: random.randint(2, 4) for m in range(1, M + 1)}
    
    teacher_subjects = {t: set() for t in range(1, T + 1)}
    subject_teachers_count = [] 
    
    # --- BƯỚC 1: PHÂN CÔNG GIÁO VIÊN DẠY MÔN GÌ ---
    # Cấy Quy luật 80/20 (Pareto) cho Adversarial
    if dist_name == "Adversarial":
        teacher_weights = np.ones(T)
        super_teachers_count = max(1, int(T * 0.2)) # 20% Super Teachers
        super_teachers_idx = random.sample(range(T), super_teachers_count)
        for idx in super_teachers_idx:
            teacher_weights[idx] = 15.0 # Trọng số x15 lần
    else:
        teacher_weights = np.ones(T) # Các phân phối khác công bằng 100%
        
    teacher_probs = teacher_weights / teacher_weights.sum()

    for m in range(1, M + 1):
        num_teachers = get_num_teachers_for_subject(dist_name, T)
        subject_teachers_count.append(num_teachers)
        
        # Bốc thăm có trọng số
        assigned_teachers_idx = np.random.choice(
            range(T), 
            size=num_teachers, 
            replace=False, 
            p=teacher_probs
        )
        for idx in assigned_teachers_idx:
            teacher_subjects[idx + 1].add(m)
            
    # --- BƯỚC 1.5: CẤY SIÊU BẪY ADVERSARIAL (NÚT THẮT CỔ CHAI) ---
    boss_subject = 1
    boss_teacher = 1
    if dist_name == "Adversarial" and M >= 5:
        # Xóa môn 1 khỏi tất cả giáo viên khác
        for t in range(1, T + 1):
            if boss_subject in teacher_subjects[t]:
                teacher_subjects[t].remove(boss_subject)
        # Chỉ duy nhất Giáo viên 1 được dạy môn 1
        teacher_subjects[boss_teacher].add(boss_subject)
        subject_teachers_count[boss_subject - 1] = 1 # Cập nhật cho biểu đồ
            
    if base_size == 1000:
        plot_data[dist_name]["teachers_per_subject"].extend(subject_teachers_count)
        plot_data[dist_name]["subjects_per_teacher"].extend([len(subs) for subs in teacher_subjects.values()])

    # --- BƯỚC 2: PHÂN CÔNG LỚP HỌC MÔN GÌ ---
    class_subjects = {n: set() for n in range(1, N + 1)}
    for n in range(1, N + 1):
        current_slots = 0
        available_subjects = list(range(1, M + 1))
        random.shuffle(available_subjects)
        
        # Ép mỗi lớp học không quá 55 tiết VÀ không quá 15 môn học/kỳ
        for m in available_subjects:
            if current_slots + d[m] <= 55 and len(class_subjects[n]) < 15:
                class_subjects[n].add(m)
                current_slots += d[m]
            else:
                if current_slots > 30 or len(class_subjects[n]) >= 15: 
                    break
        if len(class_subjects[n]) == 0:
            class_subjects[n].add(random.randint(1, M))
            
    # --- BƯỚC 2.5: ĐẨY LỚP NẠN NHÂN VÀO BẪY ---
    if dist_name == "Adversarial" and M >= 5:
        # Tính tối đa Giáo viên 1 có thể dạy bao nhiêu lớp
        max_classes = int((60 // d[boss_subject]) * 0.9) 
        victim_classes = random.sample(range(1, N + 1), min(N, max_classes))
        
        for n in victim_classes:
            if boss_subject not in class_subjects[n]:
                # Xóa bớt 1 môn để nhét Môn Độc Quyền vào
                if len(class_subjects[n]) > 0:
                    class_subjects[n].pop()
                class_subjects[n].add(boss_subject)

    # --- BƯỚC 3: GHI FILE ---
    filename = os.path.join(output_dir, f"{dist_name}_{N}_{T}_{M}.txt")
    with open(filename, 'w') as f:
        f.write(f"{T} {N} {M}\n")
        
        for n in range(1, N + 1):
            subs = list(class_subjects[n])
            f.write(" ".join(map(str, subs)) + " 0\n")
            
        for t in range(1, T + 1):
            subs = list(teacher_subjects[t])
            if not subs:
                f.write("0\n")
            else:
                f.write(" ".join(map(str, subs)) + " 0\n")
                
        f.write(" ".join(str(d[m]) for m in range(1, M + 1)) + "\n")

# --- THỰC THI CHÍNH ---
print("Bắt đầu sinh bộ dữ liệu Test Siêu Thực Tế (Kèm Bẫy Adversarial)...")
for dist in distributions:
    for size in test_sizes:
        generate_and_save_test(dist, size)
    print(f" -> Đã sinh xong 10 test cho phân phối: {dist}")

# --- VẼ BIỂU ĐỒ (VISUALIZATION) ---
print("\nĐang phân tích và vẽ biểu đồ phân phối...")
fig, axes = plt.subplots(2, 5, figsize=(22, 9))
fig.suptitle(f"Phân tích Mật độ Ràng buộc Thực tế (Tại mốc N=1000 lớp)", fontsize=18, fontweight='bold')

for i, dist in enumerate(distributions):
    axes[0, i].hist(plot_data[dist]["teachers_per_subject"], bins=20, color='#2874A6', edgecolor='black', alpha=0.85)
    axes[0, i].set_title(f"{dist}\nGiáo viên / Môn học", fontsize=12)
    axes[0, i].set_xlabel("Số lượng Giáo viên có thể dạy")
    axes[0, i].set_ylabel("Tần suất (Môn học)")
    axes[0, i].grid(axis='y', linestyle='--', alpha=0.7)
    
    axes[1, i].hist(plot_data[dist]["subjects_per_teacher"], bins=20, color='#117A65', edgecolor='black', alpha=0.85)
    axes[1, i].set_title(f"Môn học / Giáo viên", fontsize=12)
    axes[1, i].set_xlabel("Số môn học được phân công")
    axes[1, i].set_ylabel("Tần suất (Giáo viên)")
    axes[1, i].grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
image_dir = os.path.join("Figure", "Data")
os.makedirs(image_dir, exist_ok=True)
image_path = os.path.join(image_dir, "Data_Distributions_Analysis_V2.png")
plt.savefig(image_path, dpi=300, bbox_inches='tight')
print(f"Hoàn tất! Các file đã được lưu chuẩn xác vào thư mục 'Datasets'.")
print(f"Hãy mở file ảnh '{image_path}' để xem siêu bẫy Adversarial làm biến dạng biểu đồ như thế nào nhé!")