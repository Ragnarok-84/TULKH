"""
Memetic Algorithm v4 — Cải tiến toàn diện so với v3
=====================================================
Các thay đổi chính:

  [FIX 1] O(1) Blocker Lookup
    - Thêm slot_owner_cls[slot] và slot_owner_tch[slot] vào Schedule
    - Ejection tìm blocker không còn duyệt toàn bộ assignments → O(1)

  [FIX 2] Crossover thực sự ngẫu nhiên
    - Thay vì luôn ưu tiên p1, xáo ngẫu nhiên giữa p1/p2 theo từng gene
    - Tăng đa dạng quần thể, thoát local optima nhanh hơn

  [FIX 3] Adaptive Mutation strength
    - n_remove tỷ lệ theo kích thước bài (min 3, max ~5% lớp-môn)
    - Đủ mạnh để phá vỡ local optima trên bài lớn

  [FIX 4] Time-budget split: 15% init / 85% GA loop
    - Local search không ăn hết time_limit trước khi GA kịp chạy
    - Init timeout → dùng quần thể hiện tại, vẫn chạy GA với budget còn lại

  [FIX 5] LS Ejection giới hạn blocker search theo slot_owner
    - Không duyệt assignments dict nữa → nhanh hơn 10-100x trên bài lớn

  [FIX 6] Time limit hợp lý hơn cho từng bộ dataset
    - Bài nhỏ (≤500 lớp-môn): 30s thay vì 300s (greedy ~0.01s)
    - Bài lớn (>5000): 600s
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


# ─── Schedule v4: O(1) blocker lookup ────────────────────────────────────────

class Schedule:
    """
    Mở rộng từ v3:
      slot_owner_cls[cls][slot] = (cls, crs) hoặc None  → ai đang chiếm slot này của lớp cls
      slot_owner_tch[tch][slot] = (cls, crs) hoặc None  → ai đang chiếm slot này của GV tch

    Nhờ đó ejection tìm blocker chỉ cần tra O(dur) ≤ O(6) thay vì O(|assignments|).
    """
    __slots__ = ('class_busy', 'teacher_busy', 'assignments',
                 'slot_owner_cls', 'slot_owner_tch', 'N', 'T')

    def __init__(self, N, T):
        self.N = N
        self.T = T
        self.class_busy      = [bytearray(TOTAL) for _ in range(N + 1)]
        self.teacher_busy    = [bytearray(TOTAL) for _ in range(T + 1)]
        # O(1) lookup: slot_owner_cls[cls][slot] = key=(cls,crs) or None
        self.slot_owner_cls  = [[None] * TOTAL for _ in range(N + 1)]
        self.slot_owner_tch  = [[None] * TOTAL for _ in range(T + 1)]
        self.assignments     = {}   # (cls, crs) → (tch, ss)

    def copy(self):
        s = Schedule.__new__(Schedule)
        s.N = self.N
        s.T = self.T
        s.class_busy     = [bytearray(b) for b in self.class_busy]
        s.teacher_busy   = [bytearray(b) for b in self.teacher_busy]
        s.slot_owner_cls = [list(row) for row in self.slot_owner_cls]
        s.slot_owner_tch = [list(row) for row in self.slot_owner_tch]
        s.assignments    = dict(self.assignments)
        return s

    def can_place(self, cls, crs, tch, ss, dur):
        """O(dur) ≤ O(6)"""
        cb = self.class_busy[cls]
        tb = self.teacher_busy[tch]
        for k in range(dur):
            slot = ss - 1 + k
            if cb[slot] or tb[slot]:
                return False
        return True

    def place(self, cls, crs, tch, ss, dur):
        key = (cls, crs)
        cb  = self.class_busy[cls]
        tb  = self.teacher_busy[tch]
        oc  = self.slot_owner_cls[cls]
        ot  = self.slot_owner_tch[tch]
        for k in range(dur):
            slot      = ss - 1 + k
            cb[slot]  = 1
            tb[slot]  = 1
            oc[slot]  = key
            ot[slot]  = key
        self.assignments[key] = (tch, ss)

    def remove(self, cls, crs, tch, ss, dur):
        key = (cls, crs)
        cb  = self.class_busy[cls]
        tb  = self.teacher_busy[tch]
        oc  = self.slot_owner_cls[cls]
        ot  = self.slot_owner_tch[tch]
        for k in range(dur):
            slot      = ss - 1 + k
            cb[slot]  = 0
            tb[slot]  = 0
            oc[slot]  = None
            ot[slot]  = None
        del self.assignments[key]

    # [FIX 1] Tìm blockers O(dur) thay vì O(|assignments|)
    def find_blockers(self, cls, crs, tch, ss, dur):
        """
        Trả về set các key (bc, br) đang cản slot [ss, ss+dur) của cls hoặc tch.
        Chỉ duyệt dur ≤ 6 slot → O(6).
        """
        blockers = set()
        oc = self.slot_owner_cls[cls]
        ot = self.slot_owner_tch[tch]
        for k in range(dur):
            slot = ss - 1 + k
            if oc[slot] is not None:
                blockers.add(oc[slot])
            if ot[slot] is not None:
                blockers.add(ot[slot])
        return blockers

    def __len__(self):
        return len(self.assignments)


# ─── Solver ───────────────────────────────────────────────────────────────────

class MemeticSolver:
    def __init__(self, T, N, M, class_courses, teacher_courses, durations,
                 pop_size=None, generations=None, ls_iters=30,
                 mutation_rate=0.30, tournament_k=3, seed=42,
                 time_limit=None, verbose=True):
        self.T = T; self.N = N; self.M = M
        self.class_courses   = class_courses
        self.teacher_courses = teacher_courses
        self.durations       = durations
        self.ls_iters        = ls_iters
        self.mutation_rate   = mutation_rate
        self.tournament_k    = tournament_k
        self.verbose         = verbose
        self.time_limit      = time_limit
        self._start_time     = None
        self._init_deadline  = None   # [FIX 4] deadline riêng cho init
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
            if total <= 300:     pop_size = 30
            elif total <= 1000:  pop_size = 20
            elif total <= 3000:  pop_size = 12
            elif total <= 8000:  pop_size = 6
            else:                pop_size = 4
        if generations is None:
            generations = 9999  # timeout sẽ dừng sớm

        self.pop_size    = pop_size
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

        # [FIX 3] Adaptive mutation: tỷ lệ theo kích thước bài
        self._mutation_n = max(3, min(total // 20, 50))

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

    # ── Local Search v4 ───────────────────────────────────────────────────────
    # [FIX 5] Ejection dùng find_blockers O(6) thay vì duyệt assignments O(n)

    def local_search(self, sched, deadline=None):
        """
        deadline: time.perf_counter() tuyệt đối. Nếu None → không giới hạn.
        """
        sched    = sched.copy()
        assigned = set(sched.assignments.keys())

        for _iter in range(self.ls_iters):
            if deadline and time.perf_counter() >= deadline:
                break

            unassigned = [cc for cc in self.heuristic_order if cc not in assigned]
            if not unassigned:
                break
            improved = False

            # ── Phase A: Fill trực tiếp ──────────────────────────────────────
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

            # ── Phase B: Ejection Chain (O(1) blocker) ───────────────────────
            still = [cc for cc in self.heuristic_order if cc not in assigned]

            for (cls, crs) in still[:20]:   # giới hạn để tránh quá tốn time
                if deadline and time.perf_counter() >= deadline:
                    break
                dur      = self.durations[crs]
                teachers = self.teacher_order.get(crs, [])
                slots    = self.valid_slots.get(crs, [])
                ejected  = False

                for t in teachers:
                    if ejected: break
                    for s in slots:
                        if sched.can_place(cls, crs, t, s, dur):
                            sched.place(cls, crs, t, s, dur)
                            assigned.add((cls, crs))
                            improved = True
                            ejected  = True
                            break

                        # [FIX 5] O(6) blocker lookup
                        blockers = sched.find_blockers(cls, crs, t, s, dur)
                        if len(blockers) != 1:
                            continue

                        blocker     = next(iter(blockers))
                        bc, br      = blocker
                        bt, bs      = sched.assignments[blocker]
                        bd          = self.durations[br]

                        # Thử dời blocker sang slot khác
                        sched.remove(bc, br, bt, bs, bd)
                        moved   = False
                        new_ss  = None
                        for new_s in self.valid_slots.get(br, []):
                            if new_s == bs: continue
                            if sched.can_place(bc, br, bt, new_s, bd):
                                sched.place(bc, br, bt, new_s, bd)
                                moved  = True
                                new_ss = new_s
                                break

                        if moved and sched.can_place(cls, crs, t, s, dur):
                            sched.place(cls, crs, t, s, dur)
                            assigned.add((cls, crs))
                            improved = True
                            ejected  = True
                            break
                        else:
                            # Rollback blocker
                            if moved:
                                sched.remove(bc, br, bt, new_ss, bd)
                            sched.place(bc, br, bt, bs, bd)

            if not improved:
                break

        return sched

    # ── Crossover v4 ─────────────────────────────────────────────────────────
    # [FIX 2] Xáo ngẫu nhiên gene từ p1/p2 thay vì luôn ưu tiên p1

    def crossover(self, p1, p2):
        child = self._empty_schedule()

        # Shuffle thứ tự nguồn gene: 50% lấy p1 trước, 50% p2 trước
        if random.random() < 0.5:
            primary, secondary = p1, p2
        else:
            primary, secondary = p2, p1

        for (cls, crs) in self.heuristic_order:
            key = (cls, crs)
            # Thử lấy gene từ nguồn ưu tiên, fallback sang nguồn còn lại
            for src in (primary, secondary):
                if key not in src.assignments:
                    continue
                tch, ss = src.assignments[key]
                dur = self.durations[crs]
                if child.can_place(cls, crs, tch, ss, dur):
                    child.place(cls, crs, tch, ss, dur)
                    break
        return child

    # ── Mutation v4 ──────────────────────────────────────────────────────────
    # [FIX 3] n_remove adaptive, đủ mạnh để thoát local optima bài lớn

    def mutate(self, sched):
        sched = sched.copy()
        if not sched.assignments:
            return sched
        n_remove = random.randint(
            max(1, self._mutation_n // 2),
            self._mutation_n
        )
        n_remove = min(n_remove, len(sched.assignments))
        keys = random.sample(list(sched.assignments.keys()), n_remove)
        for (cls, crs) in keys:
            tch, ss = sched.assignments[(cls, crs)]
            sched.remove(cls, crs, tch, ss, self.durations[crs])
        return sched

    # ── Tournament select ─────────────────────────────────────────────────────

    def select(self, pop):
        cands = random.sample(pop, min(self.tournament_k, len(pop)))
        return max(cands, key=len)

    # ── Timeout helpers ───────────────────────────────────────────────────────

    def _elapsed(self):
        return time.perf_counter() - self._start_time

    def _timed_out(self):
        if self.time_limit is None:
            return False
        return self._elapsed() >= self.time_limit

    def _remaining(self):
        if self.time_limit is None:
            return float('inf')
        return max(0.0, self.time_limit - self._elapsed())

    # ── Main loop ────────────────────────────────────────────────────────────

    def solve(self):
        self._start_time = time.perf_counter()
        total = len(self.all_cc)
        log   = self._log

        log(f"T={self.T} GV | N={self.N} lớp | M={self.M} môn | {total} lớp-môn")
        log(f"pop={self.pop_size} | ls_iters={self.ls_iters} | mut_n={self._mutation_n}")

        # [FIX 4] Dành tối đa 15% time_limit cho init, 85% cho GA
        if self.time_limit:
            init_budget = self.time_limit * 0.15
        else:
            init_budget = None

        log("Khởi tạo quần thể...")
        pop = []

        # Cá thể 0: greedy thuần, không deadline (nhanh thôi)
        ind0 = self.make_individual(noise=0.0)
        if init_budget:
            dl = self._start_time + init_budget * 0.5   # 7.5% cho LS của ind0
        else:
            dl = None
        pop.append(self.local_search(ind0, deadline=dl))

        for i in range(1, self.pop_size):
            if init_budget and self._elapsed() >= init_budget:
                log(f"  Init timeout sau {i} cá thể, sang GA loop")
                break
            noise = 0.1 + 0.6 * (i / self.pop_size)
            ind   = self.make_individual(noise=noise)
            if init_budget:
                # chia đều budget còn lại cho các cá thể còn lại
                remaining_init = init_budget - self._elapsed()
                per_ind        = remaining_init / (self.pop_size - i) * 0.8
                dl             = time.perf_counter() + per_ind
            else:
                dl = None
            pop.append(self.local_search(ind, deadline=dl))

        if not pop:
            pop = [self.make_individual(noise=0.0)]

        best     = max(pop, key=len)
        history  = [len(best)]
        stagnant = 0

        log(f"Init xong: best={len(best)}/{total} | {self._elapsed():.1f}s dùng")
        log(f"GA loop bắt đầu | budget còn: {self._remaining():.1f}s")

        # ── GA loop ──────────────────────────────────────────────────────────
        for gen in range(1, self.generations + 1):
            if self._timed_out():
                log(f"  Timeout tại gen {gen}, dừng")
                break

            # Time còn lại: dành ít nhất 1 LS call nữa
            if self._remaining() < 0.5:
                break

            sorted_pop = sorted(pop, key=len, reverse=True)
            new_pop    = sorted_pop[:2]   # elitism top-2

            while len(new_pop) < self.pop_size:
                if self._timed_out():
                    break
                p1    = self.select(pop)
                p2    = self.select(pop)
                child = self.crossover(p1, p2)
                if random.random() < self.mutation_rate:
                    child = self.mutate(child)
                # [FIX 4] LS deadline = min(còn 0.5s, 20% remaining per child)
                n_remaining_children = self.pop_size - len(new_pop)
                per_child_budget = self._remaining() / max(n_remaining_children, 1) * 0.7
                child_dl = time.perf_counter() + per_child_budget
                child = self.local_search(child, deadline=child_dl)
                new_pop.append(child)

            pop      = new_pop
            cur_best = max(pop, key=len)
            if len(cur_best) > len(best):
                best     = cur_best
                stagnant = 0
            else:
                stagnant += 1

            history.append(len(best))

            if gen % 5 == 0 or gen <= 3:
                avg = sum(len(x) for x in pop) / len(pop)
                log(f"Gen {gen:3d}: best={len(best)}/{total} | avg={avg:.1f} | "
                    f"stagnant={stagnant} | {self._elapsed():.1f}s")

            # Partial restart khi kẹt
            if stagnant >= 15:
                log(f"  → Restart 50% quần thể (gen {gen})")
                sorted_pop = sorted(pop, key=len, reverse=True)
                keep  = sorted_pop[:self.pop_size // 2]
                fresh = []
                for i in range(self.pop_size - len(keep)):
                    if self._timed_out(): break
                    noise = 0.25 + 0.5 * random.random()
                    fresh_ind = self.make_individual(noise=noise)
                    # deadline LS ngắn khi restart để không ngốm time
                    dl = time.perf_counter() + min(2.0, self._remaining() * 0.1)
                    fresh.append(self.local_search(fresh_ind, deadline=dl))
                pop      = keep + fresh
                stagnant = 0

        log(f"Kết thúc: best={len(best)}/{total} | {self._elapsed():.2f}s")
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
SOLVER_NAME = "Memetic_Algorithm_v4"
RESULT_DIR  = os.path.join("Result", SOLVER_NAME)

# Tham số — None = tự động scale
POP_SIZE      = None
GENERATIONS   = None
LS_ITERS      = 30
MUTATION_RATE = 0.30
TOURNAMENT_K  = 3
SEED          = 42

# Chỉ chạy các subfolder này (None = tất cả)
SUBFOLDERS = None

# [FIX 6] Time limit hợp lý: greedy ~0.01-2s, MA cần ít nhất 10x để beat
# Tính theo tổng lớp-môn
TIME_LIMIT_BY_SIZE = [
    (200,   15),    # ≤ 200   lớp-môn : 15s
    (500,   30),    # ≤ 500             : 30s
    (1000,  60),    # ≤ 1000            : 60s
    (2000,  120),   # ≤ 2000            : 2 phút
    (5000,  300),   # ≤ 5000            : 5 phút
    (10000, 480),   # ≤ 10000           : 8 phút
    (float('inf'), 600),  # > 10000     : 10 phút
]

def get_time_limit(total_cc):
    for threshold, limit in TIME_LIMIT_BY_SIZE:
        if total_cc <= threshold:
            return limit
    return 600


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

        tl     = get_time_limit(total)
        start  = time.perf_counter()
        solver = MemeticSolver(
            T, N, M, class_courses, teacher_courses, durations,
            pop_size=POP_SIZE, generations=GENERATIONS, ls_iters=LS_ITERS,
            mutation_rate=MUTATION_RATE, tournament_k=TOURNAMENT_K,
            seed=SEED, time_limit=tl, verbose=True,
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

    def sort_key(fname):
        nums = re.findall(r'\d+', fname)
        return [int(n) for n in nums] if nums else [0]

    all_files = []
    for root, dirs, files in os.walk(INPUT_DIR):
        dirs.sort()
        if root == INPUT_DIR and SUBFOLDERS is not None:
            dirs[:] = [d for d in dirs if d in SUBFOLDERS]
        txts = sorted([f for f in files if f.endswith(".txt")], key=sort_key)
        for fname in txts:
            rel_dir  = os.path.relpath(root, INPUT_DIR)
            filepath = os.path.join(root, fname)
            all_files.append((rel_dir, filepath))

    if not all_files:
        print(f"Không tìm thấy file .txt trong '{INPUT_DIR}/'")
        return

    from collections import Counter
    group_count = Counter(rel for rel, _ in all_files)
    print(f"Tìm thấy {len(all_files)} file trong {len(group_count)} bộ:")
    for grp, cnt in sorted(group_count.items()):
        print(f"  {grp}: {cnt} file")
    print()

    runs          = 0
    total_time    = 0.0
    group_stats   = {}
    current_group = None

    for rel_dir, filepath in all_files:
        basename = os.path.splitext(os.path.basename(filepath))[0]

        if rel_dir != current_group:
            current_group = rel_dir
            print(f"\n{'='*60}")
            print(f"  BỘ: {rel_dir}")
            print(f"{'='*60}", flush=True)
            group_stats[rel_dir] = {'runs': 0, 'time': 0.0}

        with open(filepath, encoding='utf-8') as f:
            text = f.read()

        T, N, M, class_courses, teacher_courses, durations = parse_input(text)
        total_cc = sum(len(c) for c in class_courses)
        tl = get_time_limit(total_cc)
        print(f"  [{basename}] {total_cc} lớp-môn | limit={tl}s", flush=True)

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

        out_dir  = os.path.join(RESULT_DIR, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{SOLVER_NAME}_{basename}.txt")
        write_result(out_file, assignments, len(best), exec_time)

        rate = len(best) / total_cc * 100 if total_cc else 0
        print(f"    → Xếp: {len(best)}/{total_cc} ({rate:.1f}%) | {exec_time:.2f}s")

        runs                         += 1
        total_time                   += exec_time
        group_stats[rel_dir]['runs'] += 1
        group_stats[rel_dir]['time'] += exec_time

        # Ghi overall incremental
        _overall = os.path.join(RESULT_DIR, "Overall_Evaluation.txt")
        with open(_overall, 'w', encoding='utf-8') as _f:
            _f.write(f"Thuật toán: {SOLVER_NAME}\n")
            _f.write(f"Tổng file đã giải: {runs}\n")
            _f.write(f"Thời gian TB: {total_time/runs:.6f} giây\n\n")
            _f.write(f"{'Bộ dataset':<25} {'Số file':>8} {'TB (giây)':>12}\n")
            _f.write("-" * 48 + "\n")
            for grp, st in sorted(group_stats.items()):
                _avg = st['time'] / st['runs'] if st['runs'] else 0
                _f.write(f"{grp:<25} {st['runs']:>8} {_avg:>12.4f}\n")

    # Ghi final
    overall_file = os.path.join(RESULT_DIR, "Overall_Evaluation.txt")
    with open(overall_file, 'w', encoding='utf-8') as f:
        f.write(f"Thuật toán: {SOLVER_NAME}\n")
        f.write(f"Tổng file: {runs}\n")
        f.write(f"Thời gian TB: {total_time/runs if runs else 0:.6f} giây\n\n")
        f.write(f"{'Bộ dataset':<25} {'Số file':>8} {'TB (giây)':>12}\n")
        f.write("-" * 48 + "\n")
        for grp, stat in sorted(group_stats.items()):
            avg = stat['time'] / stat['runs'] if stat['runs'] else 0
            f.write(f"{grp:<25} {stat['runs']:>8} {avg:>12.4f}\n")

    print(f"\n{'='*60}")
    print(f"Hoàn thành! {runs} file | TB: {total_time/runs if runs else 0:.2f}s/bài")
    print(f"Kết quả: {RESULT_DIR}/")


if __name__ == "__main__":
    main()