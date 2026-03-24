import os
import time
import re
import matplotlib.pyplot as plt
from collections import defaultdict
from ortools.sat.python import cp_model

INPUT_DIR = "Datasets"
SOLVER_NAME = "CPSAT"
RESULT_DIR = os.path.join("Result", SOLVER_NAME)
TIME_LIMIT_SEC = 1000

os.makedirs(RESULT_DIR, exist_ok=True)

def read_testcase(filepath):
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    T, N, M = map(int, lines[0].split())
    class_reqs = {n: list(map(int, lines[n].split()))[:-1] for n in range(1, N + 1)}
    offset = N + 1
    teacher_caps = {t: (list(map(int, lines[offset + t - 1].split()))[:-1] if lines[offset + t - 1] != "0" else []) for t in range(1, T + 1)}
    durations = list(map(int, lines[offset + T].split()))
    d = {m: durations[m - 1] for m in range(1, M + 1)}
    return T, N, M, class_reqs, teacher_caps, d

def get_valid_slots(duration):
    return [day * 12 + session * 6 + start for day in range(5) for session in range(2) for start in range(6 - duration + 1)]

def write_result(filename, assignments, obj_val, exec_time, status):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{len(assignments)}\n")
        for n, m, u, t in assignments:
            f.write(f"{n} {m} {u} {t}\n")
        f.write(f"Điểm tối ưu: {obj_val}\n")
        f.write(f"Thời gian: {exec_time:.4f} giây\n")
        f.write(f"Trạng thái: {status}\n")

print(f"Bắt đầu chạy bộ giải {SOLVER_NAME}...")
runs, opt_count, total_time = 0, 0, 0
times_by_size = defaultdict(list)

for file in [f for f in os.listdir(INPUT_DIR) if f.endswith(".txt")]:
    filepath = os.path.join(INPUT_DIR, file)
    basename = os.path.splitext(file)[0]
    nums = re.findall(r'\d+', basename)
    size = int(nums[0]) if nums else 0
    
    T, N, M, class_reqs, teacher_caps, d = read_testcase(filepath)
    
    print(f"Đang giải {basename}...")
    model = cp_model.CpModel()
    teachers_for_m = defaultdict(list)
    for t, caps in teacher_caps.items():
        for m in caps: teachers_for_m[m].append(t)
            
    X = {}
    for n, reqs in class_reqs.items():
        for m in reqs:
            for t in teachers_for_m[m]:
                for u in get_valid_slots(d[m]):
                    X[(n, m, u, t)] = model.NewBoolVar(f'x_{n}_{m}_{u}_{t}')
                    
    for n, reqs in class_reqs.items():
        for m in reqs:
            model.AddAtMostOne([X[(n, m, u, t)] for t in teachers_for_m[m] for u in get_valid_slots(d[m])])
            
    for n in range(1, N + 1):
        for s in range(60):
            overlap = [X[(n, m, u, t)] for m in class_reqs[n] for t in teachers_for_m[m] for u in get_valid_slots(d[m]) if u <= s < u + d[m]]
            if overlap: model.AddAtMostOne(overlap)
                
    for t in range(1, T + 1):
        for s in range(60):
            overlap = [X[(n, m, u, t)] for n, reqs in class_reqs.items() for m in reqs if m in teacher_caps[t] for u in get_valid_slots(d[m]) if u <= s < u + d[m]]
            if overlap: model.AddAtMostOne(overlap)
                
    model.Maximize(sum(X.values()))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = TIME_LIMIT_SEC
    
    start_time = time.time()
    status_code = solver.Solve(model)
    exec_time = time.time() - start_time
    
    status_str = "OPTIMUM" if status_code == cp_model.OPTIMAL else "FEASIBLE" if status_code == cp_model.FEASIBLE else "UNKNOWN"
    obj_val = int(solver.ObjectiveValue()) if status_code in [cp_model.OPTIMAL, cp_model.FEASIBLE] else 0
    assignments = [(n, m, u + 1, t) for (n, m, u, t), var in X.items() if solver.Value(var) == 1] if obj_val > 0 else []
    
    out_file = os.path.join(RESULT_DIR, f"{SOLVER_NAME}_{basename}.txt")
    write_result(out_file, assignments, obj_val, exec_time, status_str)
    
    runs += 1
    total_time += exec_time
    times_by_size[size].append(exec_time)
    if status_str == "OPTIMUM": opt_count += 1

# Báo cáo
with open(os.path.join(RESULT_DIR, "Overall_Evaluation.txt"), 'w', encoding='utf-8') as f:
    f.write(f"Số bài giải thành công: {runs}\n")
    f.write(f"Số bài đạt OPTIMUM: {opt_count}/{runs}\n")
    f.write(f"Thời gian trung bình: {total_time/runs if runs else 0:.4f} giây\n")

if times_by_size:
    sizes = sorted(times_by_size.keys())
    avg_times = [sum(times_by_size[sz]) / len(times_by_size[sz]) for sz in sizes]
print(f"Hoàn thành! Đã lưu kết quả tại {RESULT_DIR}")