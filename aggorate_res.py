import os
import re

# 1. Cấu hình thư mục đọc và ghi
RESULT_DIR = os.path.join("Result", "CPSAT")
SUMMARY_DIR = os.path.join("Result", "Tổng hợp")

# Tự động tạo folder "Tổng hợp" nếu nó chưa tồn tại
os.makedirs(SUMMARY_DIR, exist_ok=True)

total_files = 0
optimum_count = 0
feasible_count = 0
total_time_sec = 0.0

print(f"Đang quét và tổng hợp dữ liệu từ thư mục {RESULT_DIR}...\n")

for filename in os.listdir(RESULT_DIR):
    # Bỏ qua các file không phải .txt hoặc các file Overall cũ
    if not filename.endswith(".txt") or "Overall" in filename:
        continue

    filepath = os.path.join(RESULT_DIR, filename)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Bóc tách Trạng thái
        status_match = re.search(r'Trạng thái:\s*([A-Z]+)', content)
        status = status_match.group(1) if status_match else "UNKNOWN"
        
        # Bóc tách Thời gian và quy đổi về giây (xử lý cả giây và ms)
        time_match = re.search(r'Thời gian:\s*([\d.]+)\s*(ms|giây|s)', content, re.IGNORECASE)
        exec_time = 0.0
        if time_match:
            val = float(time_match.group(1))
            unit = time_match.group(2).lower()
            if unit == 'ms':
                exec_time = val / 1000.0
            else:
                exec_time = val
                
        # Cộng dồn thống kê
        total_files += 1
        total_time_sec += exec_time
        
        if status == "OPTIMUM":
            optimum_count += 1
        elif status == "FEASIBLE":
            feasible_count += 1

# 2. In báo cáo và lưu file vào thư mục "Tổng hợp"
if total_files > 0:
    avg_time = total_time_sec / total_files * 1000
    
    report = (
        "=========================================\n"
        "      BÁO CÁO TỔNG HỢP CP-SAT\n"
        "=========================================\n"
        f"Tổng số bài đã giải thành công: {total_files} bài\n"
        f"Số bài đạt đỉnh OPTIMUM: {optimum_count} bài\n"
        f"Số bài chạm FEASIBLE (hết giờ): {feasible_count} bài\n"
        f"Thời gian chạy trung bình: {avg_time:.4f} ms\n"
        "=========================================\n"
    )
    
    print(report)
    
    # Lưu chính xác với tên CPSAT.txt vào thư mục Tổng hợp
    final_report_path = os.path.join(SUMMARY_DIR, "CPSAT.txt")
    with open(final_report_path, 'w', encoding='utf-8') as f:
        f.write(report)
        
    print(f"Tuyệt vời! Đã lưu file báo cáo tổng quát tại: {final_report_path}")
else:
    print(f"Không tìm thấy file kết quả nào trong {RESULT_DIR}. Hãy kiểm tra lại!")