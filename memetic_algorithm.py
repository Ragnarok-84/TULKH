"""
Memetic Algorithm v3 — O(d) conflict check với indexed schedule
================================================================
Cải tiến tốc độ so với v2:
  - can_assign: O(|chrom|) → O(duration) ≤ O(6)  ← thay đổi lớn nhất
  - Dùng class Schedule (2 mảng bool 60 slot) thay vì duyệt dict
  - Crossover/LocalSearch đều thao tác trên Schedule object
  - Pop_size và Generations tự động scale theo kích thước bài toán
"""

import random
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


# ─── Đọc input ────────────────────────────────────────────────────────────────

def parse_input(text):
    tokens = text.split()
    idx = 0
    def nx():
        nonlocal idx
        v = int(tokens[idx]); idx += 1; return v

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


# ─── Schedule: bảng tra cứu O(1) per slot ────────────────────────────────────

class Schedule:
    """
    Giữ 2 mảng bool[60]:
      class_busy[cls][slot]    = True nếu lớp cls bận slot đó
      teacher_busy[tch][slot]  = True nếu GV tch bận slot đó
    Tất cả index 0-based nội bộ, slot 1-based từ ngoài vào.
    """
    __slots__ = ('class_busy', 'teacher_busy', 'assignments', 'N', 'T')

    def __init__(self, N, T):
        self.N = N
        self.T = T
        # dùng bytearray thay list[bool] → nhỏ hơn và copy nhanh hơn
        self.class_busy   = [bytearray(TOTAL) for _ in range(N + 1)]
        self.teacher_busy = [bytearray(TOTAL) for _ in range(T + 1)]
        # assignments[(cls, crs)] = (tch, ss)
        self.assignments  = {}

    def copy(self):
        s = Schedule.__new__(Schedule)
        s.N = self.N
        s.T = self.T
        s.class_busy   = [bytearray(b) for b in self.class_busy]
        s.teacher_busy = [bytearray(b) for b in self.teacher_busy]
        s.assignments  = dict(self.assignments)
        return s

    def can_place(self, cls, crs, tch, ss, dur):
        """O(dur) ≤ O(6)"""
        cb = self.class_busy[cls]
        tb = self.teacher_busy[tch]
        for k in range(dur):
            slot = ss - 1 + k   # 0-based index
            if cb[slot] or tb[slot]:
                return False
        return True

    def place(self, cls, crs, tch, ss, dur):
        cb = self.class_busy[cls]
        tb = self.teacher_busy[tch]
        for k in range(dur):
            slot = ss - 1 + k
            cb[slot] = 1
            tb[slot] = 1
        self.assignments[(cls, crs)] = (tch, ss)

    def remove(self, cls, crs, tch, ss, dur):
        cb = self.class_busy[cls]
        tb = self.teacher_busy[tch]
        for k in range(dur):
            slot = ss - 1 + k
            cb[slot] = 0
            tb[slot] = 0
        del self.assignments[(cls, crs)]

    def __len__(self):
        return len(self.assignments)


# ─── Solver ───────────────────────────────────────────────────────────────────

class MemeticSolver:
    def __init__(self, T, N, M, class_courses, teacher_courses, durations,
                 pop_size=None, generations=None, ls_iters=20,
                 mutation_rate=0.25, tournament_k=3, seed=42,
                 time_limit=None, verbose=True):
        self.T = T; self.N = N; self.M = M
        self.class_courses   = class_courses
        self.teacher_courses = teacher_courses
        self.durations       = durations
        self.ls_iters        = ls_iters
        self.mutation_rate   = mutation_rate
        self.tournament_k    = tournament_k
        self.verbose         = verbose
        self.time_limit      = time_limit   # giây, None = không giới hạn
        self._start_time     = None
        random.seed(seed)

        # Precompute valid slots
        self.valid_slots = {m: valid_start_slots(d) for m, d in durations.items()}

        # course → list teacher (1-based)
        self.course_teachers = defaultdict(list)
        for t_idx, cs in enumerate(teacher_courses):
            for c in cs:
                self.course_teachers[c].append(t_idx + 1)

        # Tất cả lớp-môn
        self.all_cc = [
            (cls + 1, crs)
            for cls, courses in enumerate(class_courses)
            for crs in courses
        ]
        total = len(self.all_cc)

        # Auto scale tham số theo kích thước
        if pop_size is None:
            if total <= 500:    pop_size = 40
            elif total <= 2000: pop_size = 20
            elif total <= 5000: pop_size = 10
            else:               pop_size = 5   # bài rất lớn: chủ yếu dựa vào LS
        if generations is None:
            if total <= 500:    generations = 100
            elif total <= 2000: generations = 50
            elif total <= 5000: generations = 20
            else:               generations = 10

        self.pop_size   = pop_size
        self.generations = generations

        # Heuristic order: môn ít GV trước, tiết dài trước
        self.heuristic_order = sorted(
            self.all_cc,
            key=lambda x: (len(self.course_teachers[x[1]]), -durations[x[1]])
        )

        # Teacher order cho mỗi môn: GV ít môn nhất trước
        self.teacher_order = {
            crs: sorted(ts, key=lambda t: len(teacher_courses[t - 1]))
            for crs, ts in self.course_teachers.items()
        }

    # ── Tạo Schedule rỗng ────────────────────────────────────────────────────

    def _empty_schedule(self):
        return Schedule(self.N, self.T)

    # ── Khởi tạo cá thể ──────────────────────────────────────────────────────

    def make_individual(self, noise=0.0):
        sched = self._empty_schedule()
        order = list(self.heuristic_order)

        if noise > 0:
            split = int(len(order) * (1 - noise))
            tail  = order[split:]
            random.shuffle(tail)
            order = order[:split] + tail

        for (cls, crs) in order:
            teachers = self.teacher_order.get(crs, [])
            slots    = self.valid_slots.get(crs, [])
            dur      = self.durations[crs]
            if not teachers or not slots:
                continue

            if noise > 0:
                teachers = list(teachers); random.shuffle(teachers)
                slots    = list(slots);    random.shuffle(slots)

            for t in teachers:
                placed = False
                for s in slots:
                    if sched.can_place(cls, crs, t, s, dur):
                        sched.place(cls, crs, t, s, dur)
                        placed = True
                        break
                if placed:
                    break

        return sched

    # ── Local Search ─────────────────────────────────────────────────────────

    def local_search(self, sched):
        sched = sched.copy()
        assigned = set(sched.assignments.keys())

        for _ in range(self.ls_iters):
            unassigned = [cc for cc in self.heuristic_order if cc not in assigned]
            if not unassigned:
                break
            improved = False

            # Phase A: Fill
            for (cls, crs) in unassigned:
                teachers = self.teacher_order.get(crs, [])
                slots    = self.valid_slots.get(crs, [])
                dur      = self.durations[crs]
                for t in teachers:
                    for s in slots:
                        if sched.can_place(cls, crs, t, s, dur):
                            sched.place(cls, crs, t, s, dur)
                            assigned.add((cls, crs))
                            improved = True
                            break
                    else:
                        continue
                    break

            # Phase B: Ejection — thử đẩy đúng 1 kẻ cản sang slot khác
            still = [cc for cc in self.heuristic_order if cc not in assigned]
            for (cls, crs) in still[:15]:
                dur   = self.durations[crs]
                teachers = self.teacher_order.get(crs, [])
                slots    = self.valid_slots.get(crs, [])

                for t in teachers:
                    ejected = False
                    for s in slots:
                        if sched.can_place(cls, crs, t, s, dur):
                            # không cần eject, Fill đã bỏ sót → thêm thẳng
                            sched.place(cls, crs, t, s, dur)
                            assigned.add((cls, crs))
                            improved = True
                            ejected = True
                            break

                        # Tìm đúng 1 kẻ cản (class conflict hoặc teacher conflict)
                        blockers = set()
                        cb = sched.class_busy[cls]
                        tb = sched.teacher_busy[t]
                        for k in range(dur):
                            slot_idx = s - 1 + k
                            if cb[slot_idx] or tb[slot_idx]:
                                # Tìm assignment nào chiếm slot này
                                for key, (bt, bs) in sched.assignments.items():
                                    bc, br = key
                                    bd = self.durations[br]
                                    if (bc == cls or bt == t) and bs - 1 <= slot_idx < bs - 1 + bd:
                                        blockers.add(key)
                                break  # chỉ cần biết có blocker

                        if len(blockers) != 1:
                            continue

                        blocker = next(iter(blockers))
                        bc, br  = blocker
                        bt, bs  = sched.assignments[blocker]
                        bd      = self.durations[br]

                        # Thử dời blocker
                        sched.remove(bc, br, bt, bs, bd)
                        moved = False
                        for new_s in self.valid_slots.get(br, []):
                            if new_s == bs: continue
                            if sched.can_place(bc, br, bt, new_s, bd):
                                sched.place(bc, br, bt, new_s, bd)
                                moved = True
                                break

                        if moved and sched.can_place(cls, crs, t, s, dur):
                            sched.place(cls, crs, t, s, dur)
                            assigned.add((cls, crs))
                            improved = True
                            ejected  = True
                            break
                        else:
                            # Hoàn lại blocker
                            if moved:
                                # blocker đã được dời nhưng slot mới không giúp được → restore
                                new_t_val = sched.assignments.get(blocker)
                                if new_t_val:
                                    sched.remove(bc, br, bt, new_t_val[1], bd)
                            sched.place(bc, br, bt, bs, bd)

                    if ejected:
                        break

            if not improved:
                break

        return sched

    # ── Crossover ────────────────────────────────────────────────────────────

    def crossover(self, p1, p2):
        child = self._empty_schedule()
        for (cls, crs) in self.heuristic_order:
            key = (cls, crs)
            # Ưu tiên p1, fallback p2
            src = p1 if key in p1.assignments else (p2 if key in p2.assignments else None)
            if src is None: continue
            tch, ss = src.assignments[key]
            dur = self.durations[crs]
            if child.can_place(cls, crs, tch, ss, dur):
                child.place(cls, crs, tch, ss, dur)
        # Bổ sung từ p2
        for (cls, crs), (tch, ss) in p2.assignments.items():
            if (cls, crs) not in child.assignments:
                dur = self.durations[crs]
                if child.can_place(cls, crs, tch, ss, dur):
                    child.place(cls, crs, tch, ss, dur)
        return child

    # ── Mutation ─────────────────────────────────────────────────────────────

    def mutate(self, sched):
        sched = sched.copy()
        if not sched.assignments:
            return sched
        n_remove = random.randint(1, min(5, len(sched.assignments)))
        keys = random.sample(list(sched.assignments.keys()), n_remove)
        for (cls, crs) in keys:
            tch, ss = sched.assignments[(cls, crs)]
            sched.remove(cls, crs, tch, ss, self.durations[crs])
        return sched

    # ── Tournament select ─────────────────────────────────────────────────────

    def select(self, pop):
        cands = random.sample(pop, min(self.tournament_k, len(pop)))
        return max(cands, key=len)

    # ── Timeout check ─────────────────────────────────────────────────────────

    def _timed_out(self):
        if self.time_limit is None:
            return False
        return time.perf_counter() - self._start_time >= self.time_limit

    # ── Main loop ────────────────────────────────────────────────────────────

    def solve(self):
        self._start_time = time.perf_counter()
        total = len(self.all_cc)
        log   = self._log

        log(f"T={self.T} GV | N={self.N} lớp | M={self.M} môn | {total} lớp-môn")
        log(f"pop={self.pop_size} | gen={self.generations} | ls_iters={self.ls_iters}")
        log("Khởi tạo quần thể...")

        pop = []
        # Cá thể 0: thuần greedy không noise
        pop.append(self.local_search(self.make_individual(noise=0.0)))
        for i in range(1, self.pop_size):
            if self._timed_out():
                log("  Timeout khi khởi tạo, dùng quần thể hiện tại")
                break
            noise = 0.1 + 0.5 * (i / self.pop_size)
            ind   = self.make_individual(noise=noise)
            pop.append(self.local_search(ind))

        best     = max(pop, key=len)
        history  = [len(best)]
        stagnant = 0

        log(f"Gen 0: best={len(best)}/{total} | avg={sum(len(x) for x in pop)/len(pop):.1f}")

        for gen in range(1, self.generations + 1):
            if self._timed_out():
                log(f"  Timeout tại gen {gen}, dừng sớm")
                break

            sorted_pop = sorted(pop, key=len, reverse=True)
            new_pop    = sorted_pop[:2]  # elitism top-2

            while len(new_pop) < self.pop_size:
                if self._timed_out():
                    break
                p1    = self.select(pop)
                p2    = self.select(pop)
                child = self.crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                child = self.local_search(child)
                new_pop.append(child)

            pop      = new_pop
            cur_best = max(pop, key=len)
            if len(cur_best) > len(best):
                best     = cur_best
                stagnant = 0
            else:
                stagnant += 1

            history.append(len(best))

            if gen % 5 == 0 or gen == self.generations:
                avg = sum(len(x) for x in pop) / len(pop)
                elapsed = time.perf_counter() - self._start_time
                log(f"Gen {gen:3d}: best={len(best)}/{total} | avg={avg:.1f} | "
                    f"stagnant={stagnant} | {elapsed:.1f}s")

            # Partial restart khi bị kẹt
            if stagnant >= 20:
                log(f"  → Restart 50% quần thể")
                sorted_pop = sorted(pop, key=len, reverse=True)
                keep  = sorted_pop[:self.pop_size // 2]
                fresh = []
                for i in range(self.pop_size - len(keep)):
                    if self._timed_out(): break
                    noise = 0.3 + 0.4 * random.random()
                    fresh.append(self.local_search(self.make_individual(noise=noise)))
                pop      = keep + fresh
                stagnant = 0

        return best, history

    def _log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def format_output(self, sched):
        lines = [str(len(sched))]
        for (cls, crs), (tch, ss) in sorted(sched.assignments.items()):
            lines.append(f"{cls} {crs} {ss} {tch}")
        return "\n".join(lines)


# ─── Config ───────────────────────────────────────────────────────────────────

INPUT_DIR   = "Datasets"
SOLVER_NAME = "Memetic_Algorithm"
RESULT_DIR  = os.path.join("Result", SOLVER_NAME)

# Tham số — None = tự động scale theo kích thước bài
POP_SIZE      = None
GENERATIONS   = None
LS_ITERS      = 20
MUTATION_RATE = 0.25
TOURNAMENT_K  = 3
SEED          = 42

# Giới hạn thời gian theo số lớp-môn (giây/file)
# Tra theo tổng lớp-môn của file, lấy ngưỡng nhỏ nhất >= total_cc
# None ở cuối = fallback không giới hạn
TIME_LIMIT_BY_SIZE = [
    (500,    60),    # <= 500  lớp-môn : 60s
    (2000,   120),   # <= 2000 lớp-môn : 2 phút
    (5000,   300),   # <= 5000 lớp-môn : 5 phút
    (10000,  480),   # <= 10000         : 8 phút
    (float('inf'), 600),  # > 10000     : 10 phút
]

def get_time_limit(total_cc):
    for threshold, limit in TIME_LIMIT_BY_SIZE:
        if total_cc <= threshold:
            return limit
    return 600

# Chỉ chạy các subfolder trong danh sách này (None = chạy tất cả)
SUBFOLDERS = None   # vd: ["Adversarial", "Exponential"] để tách notebook


# ─── Ghi kết quả ─────────────────────────────────────────────────────────────

def write_result(filename, assignments, obj_val, exec_time, status="DONE"):
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"{len(assignments)}\n")
        for n, m, u, t in assignments:
            f.write(f"{n} {m} {u} {t}\n")
        f.write(f"Điểm tối ưu: {obj_val}\n")
        f.write(f"Thời gian: {exec_time:.6f} giây\n")
        f.write(f"Trạng thái: {status}\n")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # ── Chế độ đơn lẻ ────────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        with open(input_file, encoding='utf-8') as f:
            text = f.read()

        T, N, M, class_courses, teacher_courses, durations = parse_input(text)
        total = sum(len(c) for c in class_courses)
        print(f"T={T} GV | N={N} lớp | M={M} môn | {total} lớp-môn")

        start  = time.perf_counter()
        solver = MemeticSolver(
            T, N, M, class_courses, teacher_courses, durations,
            pop_size=POP_SIZE, generations=GENERATIONS, ls_iters=LS_ITERS,
            mutation_rate=MUTATION_RATE, tournament_k=TOURNAMENT_K,
            seed=SEED, time_limit=TIME_LIMIT, verbose=True,
        )
        best, _ = solver.solve()
        exec_time = time.perf_counter() - start

        output = solver.format_output(best)
        print("\n=== KẾT QUẢ ===")
        print(output)
        print(f"Thời gian: {exec_time:.6f} giây")

        os.makedirs(RESULT_DIR, exist_ok=True)
        basename = os.path.splitext(os.path.basename(input_file))[0]
        out_file = os.path.join(RESULT_DIR, f"{SOLVER_NAME}_{basename}.txt")
        assignments = [
            (cls, crs, ss, tch)
            for (cls, crs), (tch, ss) in sorted(best.assignments.items())
        ]
        write_result(out_file, assignments, len(best), exec_time)
        print(f"✓ Kết quả lưu: {out_file}")
        return

    # ── Chế độ batch ─────────────────────────────────────────────────────────
    os.makedirs(RESULT_DIR, exist_ok=True)
    print(f"Bắt đầu chạy bộ giải {SOLVER_NAME}...")

    # ── Thu thập tất cả .txt đệ quy trong INPUT_DIR ──────────────────────────
    def sort_key(fname):
        nums = re.findall(r'\d+', fname)
        return [int(n) for n in nums] if nums else [0]

    all_files = []   # list of (subfolder_relative, filepath)
    for root, dirs, files in os.walk(INPUT_DIR):
        dirs.sort()  # duyệt subfolder theo thứ tự alphabet
        # Lọc subfolder nếu SUBFOLDERS được chỉ định
        if root == INPUT_DIR and SUBFOLDERS is not None:
            dirs[:] = [d for d in dirs if d in SUBFOLDERS]
        txts = sorted([f for f in files if f.endswith(".txt")], key=sort_key)
        for fname in txts:
            rel_dir  = os.path.relpath(root, INPUT_DIR)   # vd: "Adversarial"
            filepath = os.path.join(root, fname)
            all_files.append((rel_dir, filepath))

    if not all_files:
        print(f"Không tìm thấy file .txt trong '{INPUT_DIR}/'")
        return

    # Thống kê số file theo từng bộ
    from collections import Counter
    group_count = Counter(rel for rel, _ in all_files)
    print(f"Tìm thấy {len(all_files)} file trong {len(group_count)} bộ dataset:")
    for grp, cnt in sorted(group_count.items()):
        print(f"  {grp}: {cnt} file")
    print()

    runs           = 0
    total_time     = 0.0
    # group_stats[subfolder] = {'runs':0, 'time':0.0}
    group_stats    = {}
    current_group  = None

    for rel_dir, filepath in all_files:
        basename = os.path.splitext(os.path.basename(filepath))[0]

        # In header khi sang bộ mới
        if rel_dir != current_group:
            current_group = rel_dir
            print(f"\n{'='*55}")
            print(f"  BỘ: {rel_dir}")
            print(f"{'='*55}", flush=True)
            group_stats[rel_dir] = {'runs': 0, 'time': 0.0}

        with open(filepath, encoding='utf-8') as f:
            text = f.read()

        T, N, M, class_courses, teacher_courses, durations = parse_input(text)
        total_cc = sum(len(c) for c in class_courses)
        tl = get_time_limit(total_cc)
        print(f"  Đang giải {basename} ({total_cc} lớp-môn | limit={tl}s)...", flush=True)

        start  = time.perf_counter()
        solver = MemeticSolver(
            T, N, M, class_courses, teacher_courses, durations,
            pop_size=POP_SIZE, generations=GENERATIONS, ls_iters=LS_ITERS,
            mutation_rate=MUTATION_RATE, tournament_k=TOURNAMENT_K,
            seed=SEED, time_limit=tl, verbose=False,
        )
        best, _ = solver.solve()
        exec_time = time.perf_counter() - start

        assignments = [
            (cls, crs, ss, tch)
            for (cls, crs), (tch, ss) in sorted(best.assignments.items())
        ]

        # Mirror cấu trúc thư mục: Result/Memetic_Algorithm/Adversarial/...
        out_dir  = os.path.join(RESULT_DIR, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{SOLVER_NAME}_{basename}.txt")
        write_result(out_file, assignments, len(best), exec_time)

        print(f"    [{basename}] Xếp: {len(best)}/{total_cc} | {exec_time:.2f}s")

        runs                          += 1
        total_time                    += exec_time
        group_stats[rel_dir]['runs']  += 1
        group_stats[rel_dir]['time']  += exec_time

    # ── Overall_Evaluation.txt: tổng kết theo từng bộ ─────────────────────
    overall_file = os.path.join(RESULT_DIR, "Overall_Evaluation.txt")
    with open(overall_file, 'w', encoding='utf-8') as f:
        f.write(f"Thuật toán: {SOLVER_NAME}\n")
        f.write(f"Tổng file giải: {runs}\n")
        f.write(f"Thời gian trung bình toàn bộ: "
                f"{total_time / runs if runs else 0:.6f} giây\n\n")
        f.write(f"{'Bộ dataset':<25} {'Số file':>8} {'TB (giây)':>12}\n")
        f.write("-" * 48 + "\n")
        for grp, stat in sorted(group_stats.items()):
            avg = stat['time'] / stat['runs'] if stat['runs'] else 0
            f.write(f"{grp:<25} {stat['runs']:>8} {avg:>12.4f}\n")
        # Overall_Evaluation riêng cho từng bộ
        for grp, stat in group_stats.items():
            grp_file = os.path.join(RESULT_DIR, grp, "Overall_Evaluation.txt")
            avg = stat['time'] / stat['runs'] if stat['runs'] else 0
            with open(grp_file, 'w', encoding='utf-8') as gf:
                gf.write(f"Bộ dataset: {grp}\n")
                gf.write(f"Số bài giải: {stat['runs']}\n")
                gf.write(f"Thời gian trung bình: {avg:.6f} giây\n")

    print(f"\n{'='*55}")
    print(f"Hoàn thành! {runs} file | TB: {total_time/runs if runs else 0:.2f}s/bài")
    print(f"Kết quả: {RESULT_DIR}/")


if __name__ == "__main__":
    main()