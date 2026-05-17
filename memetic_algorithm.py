"""
Memetic Algorithm v2 — nhúng heuristic của Greedy vào khởi tạo
================================================================
Điểm khác biệt so với v1:
  1. Khởi tạo dùng Greedy-guided (task + teacher heuristic như greedy.py)
  2. Local Search dùng Ejection Chain: đẩy lớp-môn chặn đường sang slot khác
     thay vì chỉ "thêm vào chỗ trống"
  3. Đa dạng hoá quần thể bằng perturbation mạnh hơn trước local search

Cấu trúc thời gian:
  5 ngày × 2 buổi × 6 tiết = 60 slot (1-based)
  Môn phải nằm trọn trong 1 buổi.
"""

import random
import copy
import sys
import os
import re
import time
from collections import defaultdict

DAYS, SES, PER = 5, 2, 6
TOTAL = DAYS * SES * PER  # 60

# ─── Tiện ích slot ────────────────────────────────────────────────────────────

def valid_start_slots(duration):
    slots = []
    for day in range(DAYS):
        for ses in range(SES):
            for p in range(PER - duration + 1):
                slots.append(day * SES * PER + ses * PER + p + 1)  # 1-based
    return slots

def overlap(s1, d1, s2, d2):
    return s1 < s2 + d2 and s2 < s1 + d1

# ─── Đọc input ────────────────────────────────────────────────────────────────

def parse_input(text):
    tokens = text.split()
    idx = 0
    def nx(): nonlocal idx; v = int(tokens[idx]); idx += 1; return v

    T, N, M = nx(), nx(), nx()
    class_courses = []
    for _ in range(N):
        cs = []
        while True:
            v = nx()
            if v == 0: break
            cs.append(v)
        class_courses.append(cs)
    teacher_courses = []
    for _ in range(T):
        cs = set()
        while True:
            v = nx()
            if v == 0: break
            cs.add(v)
        teacher_courses.append(cs)
    durations = {}
    for m in range(1, M + 1):
        durations[m] = nx()
    return T, N, M, class_courses, teacher_courses, durations


# ─── Chromosome ───────────────────────────────────────────────────────────────
# chrom: dict { (class_id, course_id) → (teacher_id, start_slot) }  1-based

class MemeticSolver:
    def __init__(self, T, N, M, class_courses, teacher_courses, durations,
                 pop_size=50, generations=150, ls_iters=30,
                 mutation_rate=0.2, tournament_k=3, seed=42, verbose=True):
        self.T = T; self.N = N; self.M = M
        self.class_courses = class_courses
        self.teacher_courses = teacher_courses
        self.durations = durations
        self.pop_size = pop_size
        self.generations = generations
        self.ls_iters = ls_iters
        self.mutation_rate = mutation_rate
        self.tournament_k = tournament_k
        self.verbose = verbose
        random.seed(seed)

        # Precompute valid slots cho mỗi môn
        self.valid_slots = {m: valid_start_slots(d) for m, d in durations.items()}

        # course → list of teacher_id (1-based)
        self.course_teachers = defaultdict(list)
        for t_idx, cs in enumerate(teacher_courses):
            for c in cs:
                self.course_teachers[c].append(t_idx + 1)

        # Tất cả lớp-môn
        self.all_cc = [(cls + 1, crs)
                       for cls, courses in enumerate(class_courses)
                       for crs in courses]

        # ── Heuristic order (copy từ greedy.py) ──────────────────────────────
        # Ưu tiên 1: môn ít GV dạy lên trước
        # Ưu tiên 2: trong cùng nhóm đó, môn tiết dài lên trước
        self.heuristic_order = sorted(
            self.all_cc,
            key=lambda x: (len(self.course_teachers[x[1]]), -durations[x[1]])
        )

        # Teacher order cho mỗi môn: GV biết ít môn nhất lên trước
        self.teacher_order = {}
        for crs, teachers in self.course_teachers.items():
            self.teacher_order[crs] = sorted(
                teachers,
                key=lambda t: len(teacher_courses[t - 1])
            )

    # ── Check xung đột ────────────────────────────────────────────────────────

    def can_assign(self, chrom, cls, crs, tch, ss, exclude_key=None):
        d = self.durations[crs]
        for (c2, r2), (t2, s2) in chrom.items():
            if (c2, r2) == exclude_key:
                continue
            if c2 == cls or t2 == tch:
                if overlap(ss, d, s2, self.durations[r2]):
                    return False
        return True

    # ── Khởi tạo cá thể ───────────────────────────────────────────────────────

    def make_individual(self, use_heuristic=True, noise=0.0):
        """
        noise=0.0 → thuần greedy order
        noise>0   → shuffle một phần để đa dạng hoá quần thể
        """
        chrom = {}
        order = list(self.heuristic_order) if use_heuristic else list(self.all_cc)

        if noise > 0:
            # Xáo trộn một phần: giữ nguyên top (1-noise) đầu, random phần còn lại
            split = int(len(order) * (1 - noise))
            tail = order[split:]
            random.shuffle(tail)
            order = order[:split] + tail

        for (cls, crs) in order:
            teachers = self.teacher_order.get(crs, [])
            slots = self.valid_slots.get(crs, [])
            if not teachers or not slots:
                continue

            if noise > 0:
                teachers = list(teachers)
                random.shuffle(teachers)
                slots = list(slots)
                random.shuffle(slots)

            placed = False
            for t in teachers:
                if placed: break
                for s in slots:
                    if self.can_assign(chrom, cls, crs, t, s):
                        chrom[(cls, crs)] = (t, s)
                        placed = True
                        break
        return chrom

    # ── Local Search (Ejection + Fill) ────────────────────────────────────────

    def local_search(self, chrom):
        """
        Phase A — Fill: thêm lớp-môn chưa xếp (dùng heuristic order)
        Phase B — Eject & retry: với lớp-môn chưa xếp, thử "đẩy" 1 lớp-môn
                  đang chặn sang slot khác để nhường chỗ
        """
        chrom = dict(chrom)
        assigned = set(chrom.keys())

        for _ in range(self.ls_iters):
            unassigned = [cc for cc in self.heuristic_order if cc not in assigned]
            if not unassigned:
                break
            improved = False

            # Phase A: Fill
            for (cls, crs) in unassigned:
                teachers = self.teacher_order.get(crs, [])
                slots = self.valid_slots.get(crs, [])
                for t in teachers:
                    for s in slots:
                        if self.can_assign(chrom, cls, crs, t, s):
                            chrom[(cls, crs)] = (t, s)
                            assigned.add((cls, crs))
                            improved = True
                            break
                    else:
                        continue
                    break

            # Phase B: Ejection Chain (thử đẩy kẻ cản đường)
            still_unassigned = [cc for cc in self.heuristic_order if cc not in assigned]
            for (cls, crs) in still_unassigned[:10]:  # giới hạn để không chậm
                d = self.durations[crs]
                teachers = self.teacher_order.get(crs, [])
                slots = self.valid_slots.get(crs, [])

                for t in teachers:
                    for s in slots:
                        # Tìm kẻ cản
                        blockers = [
                            (c2, r2) for (c2, r2), (t2, s2) in chrom.items()
                            if (c2 == cls or t2 == t) and overlap(s, d, s2, self.durations[r2])
                        ]
                        if len(blockers) != 1:
                            continue
                        blocker = blockers[0]
                        bt, bs = chrom[blocker]
                        bc, br = blocker

                        # Thử dời blocker sang slot khác
                        del chrom[blocker]
                        moved = False
                        for new_s in self.valid_slots.get(br, []):
                            if new_s == bs:
                                continue
                            if self.can_assign(chrom, bc, br, bt, new_s):
                                chrom[blocker] = (bt, new_s)
                                moved = True
                                break

                        if moved and self.can_assign(chrom, cls, crs, t, s):
                            chrom[(cls, crs)] = (t, s)
                            assigned.add((cls, crs))
                            improved = True
                            break
                        else:
                            chrom[blocker] = (bt, bs)  # hoàn lại
                    else:
                        continue
                    break

            if not improved:
                break

        return chrom

    # ── Crossover ─────────────────────────────────────────────────────────────

    def crossover(self, p1, p2):
        child = {}
        # Lấy từ p1 theo heuristic order (ưu tiên lớp-môn khó trước)
        for (cls, crs) in self.heuristic_order:
            key = (cls, crs)
            src = p1 if key in p1 else (p2 if key in p2 else None)
            if src is None:
                continue
            t, s = src[key]
            if self.can_assign(child, cls, crs, t, s):
                child[key] = (t, s)
        # Bổ sung từ p2
        for key, (t, s) in p2.items():
            if key not in child:
                cls, crs = key
                if self.can_assign(child, cls, crs, t, s):
                    child[key] = (t, s)
        return child

    # ── Mutation ──────────────────────────────────────────────────────────────

    def mutate(self, chrom):
        chrom = dict(chrom)
        if not chrom:
            return chrom
        # Xoá ngẫu nhiên 1-3 lớp-môn rồi để local search lấp lại
        n_remove = random.randint(1, min(3, len(chrom)))
        keys = random.sample(list(chrom.keys()), n_remove)
        for k in keys:
            del chrom[k]
        return chrom

    # ── Tournament selection ───────────────────────────────────────────────────

    def select(self, pop):
        cands = random.sample(pop, min(self.tournament_k, len(pop)))
        return max(cands, key=len)

    # ── Main loop ─────────────────────────────────────────────────────────────

    def solve(self):
        total = len(self.all_cc)
        log = self._log

        log(f"T={self.T} GV | N={self.N} lớp | M={self.M} môn | {total} lớp-môn")
        log(f"Quần thể={self.pop_size} | Thế hệ={self.generations}")
        log("Khởi tạo quần thể (greedy-guided)...")

        # Khởi tạo: 1 cá thể thuần greedy + phần còn lại có noise dần tăng
        pop = []
        pop.append(self.local_search(self.make_individual(use_heuristic=True, noise=0.0)))
        for i in range(1, self.pop_size):
            noise = 0.1 + 0.5 * (i / self.pop_size)  # noise từ 0.1 → 0.6
            ind = self.make_individual(use_heuristic=True, noise=noise)
            ind = self.local_search(ind)
            pop.append(ind)

        best = max(pop, key=len)
        history = [len(best)]
        log(f"Gen 0: best={len(best)}/{total} | avg={sum(len(x) for x in pop)/len(pop):.1f}")

        stagnant = 0
        for gen in range(1, self.generations + 1):
            sorted_pop = sorted(pop, key=len, reverse=True)
            new_pop = sorted_pop[:2]  # elitism

            while len(new_pop) < self.pop_size:
                p1 = self.select(pop)
                p2 = self.select(pop)
                child = self.crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                child = self.local_search(child)
                new_pop.append(child)

            pop = new_pop
            cur_best = max(pop, key=len)
            if len(cur_best) > len(best):
                best = cur_best
                stagnant = 0
            else:
                stagnant += 1

            history.append(len(best))

            if gen % 20 == 0 or gen == self.generations:
                avg = sum(len(x) for x in pop) / len(pop)
                log(f"Gen {gen:3d}: best={len(best)}/{total} | avg={avg:.1f} | stagnant={stagnant}")

            # Restart một phần nếu bị kẹt quá lâu
            if stagnant >= 30:
                log(f"  → Restart 50% quần thể (stagnant={stagnant})")
                sorted_pop = sorted(pop, key=len, reverse=True)
                keep = sorted_pop[:self.pop_size // 2]
                fresh = []
                for i in range(self.pop_size - len(keep)):
                    noise = 0.3 + 0.4 * random.random()
                    ind = self.make_individual(use_heuristic=True, noise=noise)
                    fresh.append(self.local_search(ind))
                pop = keep + fresh
                stagnant = 0

        return best, history

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # ── Output ────────────────────────────────────────────────────────────────

    def format_output(self, chrom):
        lines = [str(len(chrom))]
        for (cls, crs), (tch, ss) in sorted(chrom.items()):
            lines.append(f"{cls} {crs} {ss} {tch}")
        return "\n".join(lines)


# ─── Config ───────────────────────────────────────────────────────────────────

INPUT_DIR  = "Datasets"
SOLVER_NAME = "Memetic_Algorithm"
RESULT_DIR  = os.path.join("Result", SOLVER_NAME)

# Tham số MA — chỉnh tại đây
POP_SIZE       = 50
GENERATIONS    = 150
LS_ITERS       = 25
MUTATION_RATE  = 0.25
TOURNAMENT_K   = 3
SEED           = 42


# ─── Ghi kết quả (cùng định dạng với Greedy và CP-SAT) ───────────────────────

def write_result(filename, assignments, obj_val, exec_time, status="DONE"):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{len(assignments)}\n")
        for n, m, u, t in assignments:
            f.write(f"{n} {m} {u} {t}\n")
        f.write(f"Điểm tối ưu: {obj_val}\n")
        f.write(f"Thời gian: {exec_time:.6f} giây\n")
        f.write(f"Trạng thái: {status}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Chế độ đơn lẻ: python ma_v2.py input.txt
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        with open(input_file, encoding='utf-8') as f:
            text = f.read()

        T, N, M, class_courses, teacher_courses, durations = parse_input(text)
        total = sum(len(c) for c in class_courses)
        print(f"T={T} GV | N={N} lớp | M={M} môn | {total} lớp-môn")

        start = time.perf_counter()
        solver = MemeticSolver(
            T, N, M, class_courses, teacher_courses, durations,
            pop_size=POP_SIZE, generations=GENERATIONS, ls_iters=LS_ITERS,
            mutation_rate=MUTATION_RATE, tournament_k=TOURNAMENT_K,
            seed=SEED, verbose=True,
        )
        best, _ = solver.solve()
        exec_time = time.perf_counter() - start

        output = solver.format_output(best)
        print("\n=== KẾT QUẢ ===")
        print(output)
        print(f"Thời gian: {exec_time:.6f} giây")

        # Lưu file kết quả bên cạnh script
        os.makedirs(RESULT_DIR, exist_ok=True)
        basename = os.path.splitext(os.path.basename(input_file))[0]
        out_file = os.path.join(RESULT_DIR, f"{SOLVER_NAME}_{basename}.txt")
        assignments = [
            (cls, crs, ss, tch)
            for (cls, crs), (tch, ss) in sorted(best.items())
        ]
        write_result(out_file, assignments, len(best), exec_time)
        print(f"✓ Kết quả lưu: {out_file}")
        return

    # Chế độ batch: duyệt toàn bộ Datasets/ như Greedy và CP-SAT
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"Bắt đầu chạy bộ giải {SOLVER_NAME}...")

    runs = 0
    total_time = 0.0
    times_by_size = defaultdict(list)

    txt_files = sorted(f for f in os.listdir(INPUT_DIR) if f.endswith(".txt"))
    if not txt_files:
        print(f"Không tìm thấy file .txt trong '{INPUT_DIR}/'")
        return

    for file in txt_files:
        filepath = os.path.join(INPUT_DIR, file)
        basename = os.path.splitext(file)[0]
        nums = re.findall(r'\d+', basename)
        size = int(nums[0]) if nums else 0

        with open(filepath, encoding='utf-8') as f:
            text = f.read()

        T, N, M, class_courses, teacher_courses, durations = parse_input(text)
        total_cc = sum(len(c) for c in class_courses)
        print(f"Đang giải {basename} ({total_cc} lớp-môn)...", flush=True)

        start = time.perf_counter()
        solver = MemeticSolver(
            T, N, M, class_courses, teacher_courses, durations,
            pop_size=POP_SIZE, generations=GENERATIONS, ls_iters=LS_ITERS,
            mutation_rate=MUTATION_RATE, tournament_k=TOURNAMENT_K,
            seed=SEED, verbose=False,   # tắt log chi tiết khi batch
        )
        best, _ = solver.solve()
        exec_time = time.perf_counter() - start

        assignments = [
            (cls, crs, ss, tch)
            for (cls, crs), (tch, ss) in sorted(best.items())
        ]
        out_file = os.path.join(RESULT_DIR, f"{SOLVER_NAME}_{basename}.txt")
        write_result(out_file, assignments, len(best), exec_time)

        print(f"  [{basename}] Xếp: {len(best)}/{total_cc} | {exec_time:.4f}s")

        runs += 1
        total_time += exec_time
        times_by_size[size].append(exec_time)

    # Báo cáo tổng kết (cùng định dạng CP-SAT)
    overall_file = os.path.join(RESULT_DIR, "Overall_Evaluation.txt")
    with open(overall_file, 'w', encoding='utf-8') as f:
        f.write(f"Thuật toán: {SOLVER_NAME}\n")
        f.write(f"Số bài giải thành công: {runs}\n")
        f.write(f"Thời gian trung bình: {total_time / runs if runs else 0:.6f} giây\n")

    print(f"\nHoàn thành! Đã giải {runs} file. "
          f"Thời gian TB: {total_time / runs if runs else 0:.4f}s/bài")
    print(f"Kết quả lưu tại: {RESULT_DIR}/")


if __name__ == "__main__":
    main()