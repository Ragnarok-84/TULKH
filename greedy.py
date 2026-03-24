import os
import time
import re
from collections import defaultdict

INPUT_DIR = "Datasets"
SOLVER_NAME = "Greedy_Heuristic"
RESULT_DIR = os.path.join("Result", SOLVER_NAME)

os.makedirs(RESULT_DIR, exist_ok=True)

def read_testcase(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    T, N, M = map(int, lines[0].split())
    class_reqs = {n: list(map(int, lines[n].split()))[:-1] for n in range(1, N + 1)}
    offset = N + 1
    teacher_caps = {t: (list(map(int, lines[offset + t - 1].split()))[:-1] if lines[offset + t - 1] != "0" else []) for t in range(1, T + 1)}
    durations = list(map(int, lines[offset + T].split()))
    d = {m: durations[m - 1] for m in range(1, M + 1)}
    return T, N, M, class_reqs, teacher_caps, d

def write_result(filename, assignments, obj_val, exec_time):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{len(assignments)}\n")
        for n, m, u, t in assignments:
            f.write(f"{n} {m} {u} {t}\n")
        f.write(f"Điểm tối ưu: {obj_val}\n")
        f.write(f"Thời gian: {exec_time:.6f} giây\n")
        f.write(f"Trạng thái: GREEDY_DONE\n")

print("Bắt đầu chạy Thuật toán Tham lam (Greedy + Heuristic)...")
runs = 0
total_time = 0
times_by_size = defaultdict(list)

for file in os.listdir(INPUT_DIR):
    if not file.endswith(".txt"): continue
    
    filepath = os.path.join(INPUT_DIR, file)
    basename = os.path.splitext(file)[0]
    nums = re.findall(r'\d+', basename)
    size = int(nums[0]) if nums else 0
    
    T, N, M, class_reqs, teacher_caps, d = read_testcase(filepath)
    
    # --- BẮT ĐẦU TÍNH GIỜ GREEDY ---
    start_time = time.perf_counter()
    
    # 1. Thống kê giáo viên cho từng môn
    teachers_for_m = defaultdict(list)
    for t, caps in teacher_caps.items():
        for m in caps:
            teachers_for_m[m].append(t)
            
    # 2. Tạo danh sách Nhiệm vụ (Task)
    tasks = []
    for n, reqs in class_reqs.items():
        for m in reqs:
            tasks.append((n, m))
            
    # 3. HEURISTIC: Trái tim của thuật toán
    # Ưu tiên 1: Môn càng ít GV dạy càng xếp trước (len tăng dần)
    # Ưu tiên 2: Môn số tiết d(m) càng dài càng xếp trước (-d[m] giảm dần)
    tasks.sort(key=lambda x: (len(teachers_for_m[x[1]]), -d[x[1]]))
    
    # 4. Khởi tạo Lịch biểu (False là Rảnh, True là Bận)
    class_schedule = {n: [False] * 60 for n in range(1, N + 1)}
    teacher_schedule = {t: [False] * 60 for t in range(1, T + 1)}
    assignments = []
    
    # 5. Ráp lịch
    for n, m in tasks:
        duration = d[m]
        assigned = False
        
        for t in teachers_for_m[m]:
            if assigned: break
            
            for session in range(10): # 10 buổi học trong tuần
                if assigned: break
                
                for start_idx in range(6 - duration + 1):
                    u = session * 6 + start_idx
                    
                    # Kiểm tra xem từ u đến u + duration, cả Lớp và GV có rảnh không?
                    class_free = all(not class_schedule[n][u + k] for k in range(duration))
                    teacher_free = all(not teacher_schedule[t][u + k] for k in range(duration))
                    
                    if class_free and teacher_free:
                        # Gán lịch -> Cập nhật trạng thái Bận
                        for k in range(duration):
                            class_schedule[n][u + k] = True
                            teacher_schedule[t][u + k] = True
                        
                        assignments.append((n, m, u + 1, t)) # Cộng 1 vì Output đề bài yêu cầu kíp bắt đầu từ 1
                        assigned = True
                        break
                        
    exec_time = time.perf_counter() - start_time
    # --- KẾT THÚC TÍNH GIỜ ---
    
    obj_val = len(assignments)
    out_file = os.path.join(RESULT_DIR, f"{SOLVER_NAME}_{basename}.txt")
    write_result(out_file, assignments, obj_val, exec_time)
    
    print(f"[{basename}] - Đã xếp: {obj_val}/{len(tasks)} tasks - Thời gian: {exec_time:.4f}s")
    
    runs += 1
    total_time += exec_time
    times_by_size[size].append(exec_time)

# Báo cáo tổng kết
with open(os.path.join(RESULT_DIR, "Overall_Evaluation.txt"), 'w', encoding='utf-8') as f:
    f.write(f"Thuật toán: GREEDY + HEURISTIC\n")
    f.write(f"Số bài giải thành công: {runs}\n")
    f.write(f"Thời gian trung bình: {total_time/runs if runs else 0:.6f} giây\n")

print(f"\nHoàn thành! Đã chạy mượt mà {runs} file. Thời gian trung bình: {total_time/runs:.6f} giây/bài.")