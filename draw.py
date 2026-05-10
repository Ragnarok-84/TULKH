import os
import re
import matplotlib.pyplot as plt
from collections import defaultdict

# --- CẤU HÌNH ---
BIN_SIZE = 100 

# Bảng màu giống hệt hình mẫu bài báo của bạn
SOLVERS_CONFIG = [
    {"name": "CP-SAT (Exact)", "dir": os.path.join("Result", "CPSAT"), "color": "#ff7f0e", "marker": "s"},   # Cam
    {"name": "Greedy (Heuristic)", "dir": os.path.join("Result", "Greedy_Heuristic"), "color": "#1f77b4", "marker": "o"}, # Xanh dương
]

def get_binned_data():
    solver_data = {}
    all_bins_set = set()
    
    for solver in SOLVERS_CONFIG:
        dir_path = solver['dir']
        if not os.path.exists(dir_path):
            continue
            
        binned = defaultdict(list)
        for filename in os.listdir(dir_path):
            if not filename.endswith(".txt") or "Overall" in filename or filename == "CPSAT.txt":
                continue
                
            nums = re.findall(r'\d+', filename)
            if not nums: continue
            N = int(nums[0])
            
            filepath = os.path.join(dir_path, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    time_match = re.search(r'Thời gian:\s*([\d.]+)\s*(ms|giây|s)', content, re.IGNORECASE)
                    
                    if time_match:
                        val = float(time_match.group(1))
                        unit = time_match.group(2).lower()
                        exec_time = val if unit == 'ms' else val * 1000.0
                        
                        # Gom cụm theo định dạng khoảng [100-200)
                        bin_start = (N // BIN_SIZE) * BIN_SIZE 
                        bin_end = bin_start + BIN_SIZE
                        bin_label = f"[{bin_start}-{bin_end})"
                        
                        # Lưu tuple (bin_start, bin_label) để lát nữa dễ sắp xếp
                        binned[(bin_start, bin_label)].append(exec_time)
            except Exception:
                pass

        if binned:
            # Tính trung bình thời gian
            avg_binned = {k: sum(v)/len(v) for k, v in binned.items()}
            solver_data[solver['name']] = avg_binned
            all_bins_set.update(avg_binned.keys())
            
    # Sắp xếp các mốc N tăng dần
    sorted_bins = sorted(list(all_bins_set), key=lambda x: x[0])
    return solver_data, sorted_bins

# --- LẤY DỮ LIỆU ---
print("Đang xử lý dữ liệu...")
solver_data, sorted_bins = get_binned_data()

# Tạo danh sách nhãn Trục X và các chỉ số (index) để các điểm cách đều nhau
X_labels = [b[1] for b in sorted_bins]
X_indices = list(range(len(sorted_bins)))

# --- VẼ BIỂU ĐỒ ---
# Kích thước khung hình chuẩn bài báo khoa học
plt.figure(figsize=(10, 6.5))

for solver in SOLVERS_CONFIG:
    name = solver['name']
    if name not in solver_data:
        continue
        
    data = solver_data[name]
    
    # Chỉ vẽ những điểm có dữ liệu thực tế
    y_vals = []
    x_vals = []
    for i, b in enumerate(sorted_bins):
        if b in data:
            x_vals.append(i)
            y_vals.append(data[b])
            
    if x_vals:
        plt.plot(x_vals, y_vals, 
                 label=name, 
                 color=solver['color'], 
                 marker=solver['marker'], 
                 markersize=5, 
                 linestyle='-', 
                 linewidth=1.2,
                 alpha=0.8)

# --- ĐỊNH DẠNG GIỐNG HỆT BÀI BÁO (LOG SCALE) ---
# 1. Chuyển trục Y sang thang đo Logarit
plt.yscale('log')

# 2. Xóa khoảng thừa lòi ra ở hai bên hông biểu đồ
if X_indices:
    plt.xlim(X_indices[0] - 0.2, X_indices[-1] + 0.2)

# 3. Gắn nhãn danh mục cho trục X
plt.xticks(X_indices, X_labels, rotation=45, ha='right', fontsize=10)

# 4. Bật lưới chuẩn bị (Cả lưới chính và lưới phụ mờ mờ cho logarit)
plt.grid(True, which='major', linestyle='-', linewidth=0.8, color='lightgray')
plt.grid(True, which='minor', linestyle=':', linewidth=0.5, color='lightgray')

# 5. Tiêu đề và nhãn
plt.xlabel('Kích thước bài toán (N)', fontsize=12, fontweight='bold')
plt.ylabel('Thời gian thực thi trung bình (ms)', fontsize=12, fontweight='bold')
plt.title('So sánh Thời gian thực thi (Thang đo Logarit)', fontsize=14, pad=15)

plt.legend(loc='upper left', frameon=True, fontsize=11, shadow=True)
plt.tight_layout()

output_file = 'Bieu_Do_Log_Scale_Chuan.png'
plt.savefig(output_file, dpi=300)
print(f"Đã xuất xong! Bạn mở file '{output_file}' để xem giao diện chuẩn paper nhé.")
plt.show()