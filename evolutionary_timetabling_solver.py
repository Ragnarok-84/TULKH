"""
Evolutionary Timetabling Solver — stronger-than-greedy baseline
===============================================================
Bài: Class - Course - Teacher Assignment and Timetabling

Ý tưởng chính:
- Luôn chạy smart greedy làm baseline.
- Thuật toán tiến hóa tối ưu "thứ tự xếp lớp-môn" thay vì chỉ mutate lịch đã xếp.
- Decode một chromosome bằng randomized greedy.
- Có local search phá lịch rồi xếp lại theo thứ tự ưu tiên mới.
- Trả về nghiệm tốt nhất giữa GREEDY và EVOLUTIONARY, nên kết quả KHÔNG BAO GIỜ thấp hơn greedy.

Chạy batch:
    python evolutionary_timetabling_solver.py

Chạy 1 file:
    python evolutionary_timetabling_solver.py Datasets/Uniform/xxx.txt
"""

import os
import re
import sys
import time
import random
from collections import defaultdict, Counter

DAYS = 5
SESSIONS_PER_DAY = 2
PERIODS_PER_SESSION = 6
TOTAL_SLOTS = DAYS * SESSIONS_PER_DAY * PERIODS_PER_SESSION  # 60

INPUT_DIR = "Datasets"
SOLVER_NAME = "Evolutionary_Timetabling_v1"
RESULT_DIR = os.path.join("Result", SOLVER_NAME)

# Có thể chỉnh nếu muốn chạy lâu hơn
SEED = 42
SUBFOLDERS = None  # None = chạy tất cả: Adversarial, Exponential, Gaussian, hu_stack, Poisson, Uniform...

TIME_LIMIT_BY_SIZE = [
    (200, 20),
    (500, 45),
    (1000, 90),
    (2000, 180),
    (5000, 360),
    (10000, 600),
    (float("inf"), 900),
]


def get_time_limit(total_cc):
    for threshold, limit in TIME_LIMIT_BY_SIZE:
        if total_cc <= threshold:
            return limit
    return 900


# ============================================================
# Parser / Writer
# ============================================================

def parse_input(text):
    tokens = list(map(int, text.split()))
    idx = 0

    def nx():
        nonlocal idx
        val = tokens[idx]
        idx += 1
        return val

    T, N, M = nx(), nx(), nx()

    class_courses = [[] for _ in range(N + 1)]
    for cls in range(1, N + 1):
        while True:
            c = nx()
            if c == 0:
                break
            class_courses[cls].append(c)

    teacher_courses = [set() for _ in range(T + 1)]
    for t in range(1, T + 1):
        while True:
            c = nx()
            if c == 0:
                break
            teacher_courses[t].add(c)

    durations = [0] * (M + 1)
    for m in range(1, M + 1):
        durations[m] = nx()

    return T, N, M, class_courses, teacher_courses, durations


def write_result(filename, assignments, obj_val, exec_time, status="DONE"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"{len(assignments)}\n")
        for cls, crs, start, teacher in assignments:
            f.write(f"{cls} {crs} {start} {teacher}\n")
        f.write(f"Điểm tối ưu: {obj_val}\n")
        f.write(f"Thời gian: {exec_time:.6f} giây\n")
        f.write(f"Trạng thái: {status}\n")


# ============================================================
# Schedule structure
# ============================================================

class Schedule:
    __slots__ = (
        "N", "T", "class_busy", "teacher_busy",
        "owner_class", "owner_teacher", "assignments"
    )

    def __init__(self, N, T):
        self.N = N
        self.T = T
        self.class_busy = [bytearray(TOTAL_SLOTS) for _ in range(N + 1)]
        self.teacher_busy = [bytearray(TOTAL_SLOTS) for _ in range(T + 1)]
        self.owner_class = [[None] * TOTAL_SLOTS for _ in range(N + 1)]
        self.owner_teacher = [[None] * TOTAL_SLOTS for _ in range(T + 1)]
        self.assignments = {}  # (cls, crs) -> (teacher, start_1_based)

    def copy(self):
        s = Schedule.__new__(Schedule)
        s.N = self.N
        s.T = self.T
        s.class_busy = [bytearray(x) for x in self.class_busy]
        s.teacher_busy = [bytearray(x) for x in self.teacher_busy]
        s.owner_class = [list(x) for x in self.owner_class]
        s.owner_teacher = [list(x) for x in self.owner_teacher]
        s.assignments = dict(self.assignments)
        return s

    def can_place(self, cls, crs, teacher, start, dur):
        base = start - 1
        cb = self.class_busy[cls]
        tb = self.teacher_busy[teacher]
        for k in range(dur):
            slot = base + k
            if cb[slot] or tb[slot]:
                return False
        return True

    def place(self, cls, crs, teacher, start, dur):
        key = (cls, crs)
        base = start - 1
        cb = self.class_busy[cls]
        tb = self.teacher_busy[teacher]
        oc = self.owner_class[cls]
        ot = self.owner_teacher[teacher]
        for k in range(dur):
            slot = base + k
            cb[slot] = 1
            tb[slot] = 1
            oc[slot] = key
            ot[slot] = key
        self.assignments[key] = (teacher, start)

    def remove(self, cls, crs, teacher, start, dur):
        key = (cls, crs)
        base = start - 1
        cb = self.class_busy[cls]
        tb = self.teacher_busy[teacher]
        oc = self.owner_class[cls]
        ot = self.owner_teacher[teacher]
        for k in range(dur):
            slot = base + k
            cb[slot] = 0
            tb[slot] = 0
            oc[slot] = None
            ot[slot] = None
        del self.assignments[key]

    def blockers(self, cls, teacher, start, dur):
        result = set()
        base = start - 1
        oc = self.owner_class[cls]
        ot = self.owner_teacher[teacher]
        for k in range(dur):
            slot = base + k
            if oc[slot] is not None:
                result.add(oc[slot])
            if ot[slot] is not None:
                result.add(ot[slot])
        return result

    def __len__(self):
        return len(self.assignments)


# ============================================================
# Evolutionary solver
# ============================================================

class EvolutionarySolver:
    def __init__(self, T, N, M, class_courses, teacher_courses, durations,
                 time_limit=60, seed=42, verbose=True):
        self.T = T
        self.N = N
        self.M = M
        self.class_courses = class_courses
        self.teacher_courses = teacher_courses
        self.durations = durations
        self.time_limit = time_limit
        self.verbose = verbose
        self.rng = random.Random(seed)
        self.start_time = None

        self.all_cc = []
        for cls in range(1, N + 1):
            for crs in class_courses[cls]:
                self.all_cc.append((cls, crs))
        self.total_cc = len(self.all_cc)

        self.course_teachers = defaultdict(list)
        for t in range(1, T + 1):
            for c in teacher_courses[t]:
                self.course_teachers[c].append(t)

        self.teacher_load_score = [0] * (T + 1)
        for t in range(1, T + 1):
            self.teacher_load_score[t] = len(teacher_courses[t])

        self.valid_slots = {}
        for m in range(1, M + 1):
            d = durations[m]
            slots = []
            if 1 <= d <= PERIODS_PER_SESSION:
                for day in range(DAYS):
                    for ses in range(SESSIONS_PER_DAY):
                        base = (day * SESSIONS_PER_DAY + ses) * PERIODS_PER_SESSION
                        for p in range(PERIODS_PER_SESSION - d + 1):
                            slots.append(base + p + 1)
            self.valid_slots[m] = slots

        self.base_order = sorted(
            self.all_cc,
            key=lambda x: (
                len(self.course_teachers[x[1]]),
                -self.durations[x[1]],
                -len(self.class_courses[x[0]]),
                x[0],
                x[1],
            )
        )

        self.base_teacher_order = {}
        for crs, teachers in self.course_teachers.items():
            self.base_teacher_order[crs] = sorted(
                teachers,
                key=lambda t: (self.teacher_load_score[t], t)
            )

        # Auto scale tham số
        n = self.total_cc
        if n <= 200:
            self.pop_size = 40
            self.elite = 5
        elif n <= 1000:
            self.pop_size = 32
            self.elite = 4
        elif n <= 3000:
            self.pop_size = 22
            self.elite = 3
        elif n <= 8000:
            self.pop_size = 14
            self.elite = 2
        else:
            self.pop_size = 8
            self.elite = 2

        self.tournament_k = 3
        self.mutation_rate = 0.75
        self.local_kick_rate = 0.35

    # ---------------- Time helpers ----------------

    def elapsed(self):
        return time.perf_counter() - self.start_time

    def remaining(self):
        return max(0.0, self.time_limit - self.elapsed())

    def timeout(self):
        return self.elapsed() >= self.time_limit

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # ---------------- Chromosome ----------------

    def make_chromosome(self, mode="mixed"):
        """
        Chromosome = permutation của các lớp-môn.
        mode:
          - greedy: đúng base_order
          - random: shuffle toàn bộ
          - noisy: giữ bias heuristic nhưng đảo nhiều vị trí
          - hard_first: đưa các môn khó lên đầu, nhưng có nhiễu
        """
        order = list(self.base_order)

        if mode == "greedy":
            return order

        if mode == "random":
            self.rng.shuffle(order)
            return order

        if mode == "hard_first":
            def score(cc):
                cls, crs = cc
                few_teacher = len(self.course_teachers[crs])
                dur = self.durations[crs]
                cls_degree = len(self.class_courses[cls])
                jitter = self.rng.random() * 2.5
                return few_teacher * 8 - dur * 3 - cls_degree + jitter
            order.sort(key=score)
            return order

        # noisy: gán key heuristic + random noise
        n = len(order)
        max_noise = max(3, int(n * 0.20))
        pos = {cc: i for i, cc in enumerate(order)}
        order.sort(key=lambda cc: pos[cc] + self.rng.randint(-max_noise, max_noise))
        return order

    def order_crossover(self, a, b):
        """Order crossover cho permutation."""
        n = len(a)
        if n <= 2:
            return list(a)
        i = self.rng.randint(0, n - 2)
        j = self.rng.randint(i + 1, n - 1)
        child = [None] * n
        used = set()
        for k in range(i, j + 1):
            child[k] = a[k]
            used.add(a[k])
        fill = [x for x in b if x not in used]
        p = 0
        for k in range(n):
            if child[k] is None:
                child[k] = fill[p]
                p += 1
        return child

    def mutate_order(self, order):
        """Mutation mạnh trên permutation: swap, insert, reverse, hard-promotion."""
        n = len(order)
        if n <= 1:
            return list(order)
        arr = list(order)

        # số thao tác tăng theo kích thước bài
        ops = 1 + min(12, max(1, n // 300))
        for _ in range(ops):
            r = self.rng.random()
            if r < 0.40:
                i, j = self.rng.sample(range(n), 2)
                arr[i], arr[j] = arr[j], arr[i]
            elif r < 0.70:
                i, j = self.rng.sample(range(n), 2)
                item = arr.pop(i)
                arr.insert(j, item)
            elif r < 0.90:
                i = self.rng.randint(0, n - 2)
                j = self.rng.randint(i + 1, min(n - 1, i + max(2, n // 10)))
                arr[i:j + 1] = reversed(arr[i:j + 1])
            else:
                # đưa một task khó lên sớm hơn
                tail_start = n // 3
                i = self.rng.randint(tail_start, n - 1)
                item = arr.pop(i)
                j = self.rng.randint(0, max(0, n // 4))
                arr.insert(j, item)
        return arr

    # ---------------- Decode / constructive heuristic ----------------

    def teacher_order_for_decode(self, crs, random_level):
        teachers = list(self.base_teacher_order.get(crs, []))
        if random_level <= 0:
            return teachers
        # Random nhẹ nhưng vẫn bias GV ít môn trước
        teachers.sort(key=lambda t: self.teacher_load_score[t] + self.rng.random() * random_level * 5.0)
        return teachers

    def slot_order_for_decode(self, crs, random_level):
        slots = list(self.valid_slots.get(crs, []))
        if random_level <= 0:
            return slots
        # Mix: đôi khi shuffle toàn bộ để thoát greedy slot sớm
        if self.rng.random() < random_level:
            self.rng.shuffle(slots)
        else:
            slots.sort(key=lambda s: s + self.rng.random() * random_level * 20.0)
        return slots

    def decode(self, order, random_level=0.0):
        sched = Schedule(self.N, self.T)

        for cls, crs in order:
            if (cls, crs) in sched.assignments:
                continue
            dur = self.durations[crs]
            teachers = self.teacher_order_for_decode(crs, random_level)
            slots = self.slot_order_for_decode(crs, random_level)
            placed = False
            for t in teachers:
                if placed:
                    break
                for s in slots:
                    if sched.can_place(cls, crs, t, s, dur):
                        sched.place(cls, crs, t, s, dur)
                        placed = True
                        break

        return sched

    # ---------------- Local improvement ----------------

    def direct_fill(self, sched, order, random_level=0.0):
        assigned = set(sched.assignments.keys())
        improved = False
        missing = [cc for cc in order if cc not in assigned]

        # Random hóa thứ tự missing một chút để không quay lại greedy
        if random_level > 0 and self.rng.random() < 0.5:
            self.rng.shuffle(missing)

        for cls, crs in missing:
            dur = self.durations[crs]
            teachers = self.teacher_order_for_decode(crs, random_level)
            slots = self.slot_order_for_decode(crs, random_level)
            placed = False
            for t in teachers:
                if placed:
                    break
                for s in slots:
                    if sched.can_place(cls, crs, t, s, dur):
                        sched.place(cls, crs, t, s, dur)
                        assigned.add((cls, crs))
                        improved = True
                        placed = True
                        break
        return improved

    def one_ejection_improve(self, sched, target_order, random_level=0.0, max_targets=40):
        assigned = set(sched.assignments.keys())
        missing = [cc for cc in target_order if cc not in assigned]
        if random_level > 0:
            self.rng.shuffle(missing)
        else:
            missing = missing[:]

        for cls, crs in missing[:max_targets]:
            dur = self.durations[crs]
            teachers = self.teacher_order_for_decode(crs, random_level)
            slots = self.slot_order_for_decode(crs, random_level)

            for t in teachers:
                for s in slots:
                    if sched.can_place(cls, crs, t, s, dur):
                        sched.place(cls, crs, t, s, dur)
                        return True

                    bs = sched.blockers(cls, t, s, dur)
                    if len(bs) != 1:
                        continue

                    bcls, bcrs = next(iter(bs))
                    bt, bstart = sched.assignments[(bcls, bcrs)]
                    bdur = self.durations[bcrs]

                    # Thử dời blocker sang teacher/slot khác, không chỉ cùng teacher
                    sched.remove(bcls, bcrs, bt, bstart, bdur)
                    moved = False
                    new_t = None
                    new_s = None

                    bt_order = self.teacher_order_for_decode(bcrs, random_level)
                    bs_order = self.slot_order_for_decode(bcrs, random_level)

                    for cand_t in bt_order:
                        if moved:
                            break
                        for cand_s in bs_order:
                            if cand_t == bt and cand_s == bstart:
                                continue
                            if sched.can_place(bcls, bcrs, cand_t, cand_s, bdur):
                                sched.place(bcls, bcrs, cand_t, cand_s, bdur)
                                moved = True
                                new_t, new_s = cand_t, cand_s
                                break

                    if moved and sched.can_place(cls, crs, t, s, dur):
                        sched.place(cls, crs, t, s, dur)
                        return True

                    # rollback
                    if moved:
                        sched.remove(bcls, bcrs, new_t, new_s, bdur)
                    sched.place(bcls, bcrs, bt, bstart, bdur)

        return False

    def destroy_repair(self, sched, order, remove_count, random_level=0.4):
        """Large neighborhood search: xóa một phần lịch rồi repair theo order mới."""
        sched = sched.copy()
        keys = list(sched.assignments.keys())
        if not keys:
            return sched

        remove_count = min(remove_count, len(keys))

        # Ưu tiên xóa các task thuộc cuối order hoặc các môn dễ để nhường chỗ task khó
        pos = {cc: i for i, cc in enumerate(order)}
        keys.sort(key=lambda cc: pos.get(cc, 10**9), reverse=True)
        pool = keys[:max(remove_count * 4, remove_count)]
        self.rng.shuffle(pool)
        chosen = pool[:remove_count]

        for cls, crs in chosen:
            t, s = sched.assignments[(cls, crs)]
            sched.remove(cls, crs, t, s, self.durations[crs])

        self.direct_fill(sched, order, random_level=random_level)
        return sched

    def improve_schedule(self, sched, order, budget_seconds, random_level=0.25):
        deadline = time.perf_counter() + max(0.0, budget_seconds)
        best = sched.copy()

        while time.perf_counter() < deadline:
            before = len(sched)
            self.direct_fill(sched, order, random_level=random_level)
            if len(sched) > len(best):
                best = sched.copy()

            if time.perf_counter() >= deadline:
                break

            changed = self.one_ejection_improve(sched, order, random_level=random_level)
            if len(sched) > len(best):
                best = sched.copy()

            if not changed and len(sched) == before:
                break

        return best

    # ---------------- Selection / solve ----------------

    def fitness(self, item):
        return len(item["schedule"])

    def select(self, population):
        k = min(self.tournament_k, len(population))
        sample = self.rng.sample(population, k)
        return max(sample, key=self.fitness)

    def make_individual(self, mode, random_level):
        order = self.make_chromosome(mode)
        sched = self.decode(order, random_level=random_level)
        return {"order": order, "schedule": sched}

    def solve(self):
        self.start_time = time.perf_counter()
        self.log(f"T={self.T} | N={self.N} | M={self.M} | total={self.total_cc} | limit={self.time_limit}s")

        # 1) Greedy baseline. Đây là bảo hiểm để không bao giờ thua greedy.
        greedy_order = self.make_chromosome("greedy")
        greedy_sched = self.decode(greedy_order, random_level=0.0)
        greedy_sched = self.improve_schedule(greedy_sched, greedy_order, budget_seconds=min(1.0, self.time_limit * 0.03), random_level=0.0)
        best = {"order": greedy_order, "schedule": greedy_sched}
        self.log(f"Greedy baseline: {len(greedy_sched)}/{self.total_cc}")

        if len(greedy_sched) == self.total_cc:
            self.log("Greedy đã xếp đủ 100%, không thể cải thiện thêm về K.")
            return greedy_sched, len(greedy_sched), len(greedy_sched)

        # 2) Init population đa dạng
        population = [best]
        init_budget = min(self.time_limit * 0.20, max(3.0, self.time_limit - 1.0))
        init_deadline = self.start_time + init_budget

        modes = ["noisy", "hard_first", "random"]
        while len(population) < self.pop_size and time.perf_counter() < init_deadline and not self.timeout():
            mode = modes[len(population) % len(modes)]
            rl = 0.15 + 0.70 * self.rng.random()
            ind = self.make_individual(mode, rl)
            small_budget = max(0.02, (init_deadline - time.perf_counter()) / max(1, self.pop_size - len(population)))
            ind["schedule"] = self.improve_schedule(ind["schedule"], ind["order"], small_budget * 0.7, random_level=rl)
            population.append(ind)
            if self.fitness(ind) > self.fitness(best):
                best = {"order": list(ind["order"]), "schedule": ind["schedule"].copy()}

        self.log(f"Init: pop={len(population)} | best={self.fitness(best)}/{self.total_cc} | elapsed={self.elapsed():.2f}s")

        # 3) Evolution loop
        gen = 0
        stagnant = 0
        last_best = self.fitness(best)

        while not self.timeout():
            gen += 1
            if self.remaining() < 0.2:
                break

            population.sort(key=self.fitness, reverse=True)
            new_pop = population[:self.elite]

            while len(new_pop) < self.pop_size and not self.timeout():
                p1 = self.select(population)
                p2 = self.select(population)

                if self.rng.random() < 0.85:
                    child_order = self.order_crossover(p1["order"], p2["order"])
                else:
                    child_order = list(p1["order"])

                if self.rng.random() < self.mutation_rate:
                    child_order = self.mutate_order(child_order)

                random_level = 0.10 + 0.55 * self.rng.random()
                child_sched = self.decode(child_order, random_level=random_level)

                # destroy-repair trên lịch con để tạo bước nhảy lớn
                if self.rng.random() < self.local_kick_rate:
                    rem = max(2, min(60, self.total_cc // 25))
                    child_sched = self.destroy_repair(child_sched, child_order, rem, random_level=random_level)

                # budget chia đều cho số child còn lại
                left_children = max(1, self.pop_size - len(new_pop))
                child_budget = min(3.0, self.remaining() / left_children * 0.65)
                child_sched = self.improve_schedule(child_sched, child_order, child_budget, random_level=random_level)

                child = {"order": child_order, "schedule": child_sched}
                new_pop.append(child)

                if self.fitness(child) > self.fitness(best):
                    best = {"order": list(child_order), "schedule": child_sched.copy()}
                    self.log(f"  Gen {gen}: new best {self.fitness(best)}/{self.total_cc} | {self.elapsed():.2f}s")
                    if self.fitness(best) == self.total_cc:
                        return best["schedule"], len(greedy_sched), self.fitness(best)

            population = new_pop

            cur = self.fitness(best)
            if cur > last_best:
                stagnant = 0
                last_best = cur
            else:
                stagnant += 1

            if gen <= 3 or gen % 5 == 0:
                avg = sum(self.fitness(x) for x in population) / len(population)
                self.log(f"Gen {gen}: best={cur}/{self.total_cc} | avg={avg:.1f} | stagnant={stagnant} | remain={self.remaining():.1f}s")

            # restart nếu kẹt: giữ elite + bơm cá thể random/hard_first
            if stagnant >= 10 and not self.timeout():
                population.sort(key=self.fitness, reverse=True)
                keep = population[:self.elite]
                population = keep[:]
                while len(population) < self.pop_size and not self.timeout():
                    mode = "hard_first" if self.rng.random() < 0.65 else "random"
                    rl = 0.35 + 0.60 * self.rng.random()
                    ind = self.make_individual(mode, rl)
                    budget = min(1.5, self.remaining() * 0.05)
                    ind["schedule"] = self.improve_schedule(ind["schedule"], ind["order"], budget, random_level=rl)
                    population.append(ind)
                    if self.fitness(ind) > self.fitness(best):
                        best = {"order": list(ind["order"]), "schedule": ind["schedule"].copy()}
                stagnant = 0

        evo_best = self.fitness(best)
        greedy_val = len(greedy_sched)

        # Bảo hiểm cuối cùng: không bao giờ trả nghiệm tệ hơn greedy.
        if evo_best < greedy_val:
            return greedy_sched, greedy_val, greedy_val
        return best["schedule"], greedy_val, evo_best

    def format_assignments(self, sched):
        return [
            (cls, crs, start, teacher)
            for (cls, crs), (teacher, start) in sorted(sched.assignments.items())
        ]


# ============================================================
# Validation
# ============================================================

def validate_solution(T, N, M, class_courses, teacher_courses, durations, assignments):
    seen = set()
    class_busy = [bytearray(TOTAL_SLOTS) for _ in range(N + 1)]
    teacher_busy = [bytearray(TOTAL_SLOTS) for _ in range(T + 1)]

    class_req_set = [set() for _ in range(N + 1)]
    for cls in range(1, N + 1):
        class_req_set[cls] = set(class_courses[cls])

    for cls, crs, start, teacher in assignments:
        if not (1 <= cls <= N and 1 <= crs <= M and 1 <= teacher <= T):
            return False, f"ID ngoài phạm vi: {(cls, crs, start, teacher)}"
        if crs not in class_req_set[cls]:
            return False, f"Lớp {cls} không yêu cầu môn {crs}"
        if crs not in teacher_courses[teacher]:
            return False, f"GV {teacher} không dạy được môn {crs}"
        if (cls, crs) in seen:
            return False, f"Trùng lớp-môn {(cls, crs)}"
        seen.add((cls, crs))

        dur = durations[crs]
        if dur < 1 or dur > PERIODS_PER_SESSION:
            return False, f"duration không hợp lệ môn {crs}: {dur}"
        if start < 1 or start > TOTAL_SLOTS:
            return False, f"start ngoài phạm vi: {start}"
        # không được vắt qua buổi học
        start0 = start - 1
        if start0 % PERIODS_PER_SESSION + dur > PERIODS_PER_SESSION:
            return False, f"Môn {(cls, crs)} vắt qua buổi: start={start}, dur={dur}"

        for k in range(dur):
            slot = start0 + k
            if class_busy[cls][slot]:
                return False, f"Trùng lịch lớp {cls} tại slot {slot + 1}"
            if teacher_busy[teacher][slot]:
                return False, f"Trùng lịch GV {teacher} tại slot {slot + 1}"
            class_busy[cls][slot] = 1
            teacher_busy[teacher][slot] = 1

    return True, "OK"


# ============================================================
# Main modes
# ============================================================

def solve_one_file(filepath, verbose=True):
    with open(filepath, encoding="utf-8") as f:
        text = f.read()

    T, N, M, class_courses, teacher_courses, durations = parse_input(text)
    total_cc = sum(len(class_courses[cls]) for cls in range(1, N + 1))
    tl = get_time_limit(total_cc)

    start = time.perf_counter()
    solver = EvolutionarySolver(
        T, N, M, class_courses, teacher_courses, durations,
        time_limit=tl, seed=SEED, verbose=verbose
    )
    best_sched, greedy_val, evo_val = solver.solve()
    exec_time = time.perf_counter() - start

    assignments = solver.format_assignments(best_sched)
    ok, msg = validate_solution(T, N, M, class_courses, teacher_courses, durations, assignments)
    status = "VALID" if ok else f"INVALID: {msg}"

    return {
        "assignments": assignments,
        "obj": len(assignments),
        "greedy_baseline": greedy_val,
        "evo_best": evo_val,
        "total_cc": total_cc,
        "time": exec_time,
        "status": status,
    }


def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    # Chạy 1 file
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        result = solve_one_file(filepath, verbose=True)
        basename = os.path.splitext(os.path.basename(filepath))[0]
        out_file = os.path.join(RESULT_DIR, f"{SOLVER_NAME}_{basename}.txt")
        write_result(out_file, result["assignments"], result["obj"], result["time"], result["status"])

        print("\n=== RESULT ===")
        print(f"File: {filepath}")
        print(f"Greedy baseline: {result['greedy_baseline']}/{result['total_cc']}")
        print(f"Evolutionary:     {result['obj']}/{result['total_cc']}")
        print(f"Status:           {result['status']}")
        print(f"Time:             {result['time']:.3f}s")
        print(f"Saved:            {out_file}")
        return

    # Chạy batch toàn bộ Datasets
    def sort_key(fname):
        nums = re.findall(r"\d+", fname)
        return [int(x) for x in nums] if nums else [0]

    all_files = []
    for root, dirs, files in os.walk(INPUT_DIR):
        dirs.sort()
        if root == INPUT_DIR and SUBFOLDERS is not None:
            dirs[:] = [d for d in dirs if d in SUBFOLDERS]
        txts = sorted([f for f in files if f.endswith(".txt")], key=sort_key)
        for fname in txts:
            rel_dir = os.path.relpath(root, INPUT_DIR)
            all_files.append((rel_dir, os.path.join(root, fname)))

    if not all_files:
        print(f"Không tìm thấy file .txt trong {INPUT_DIR}")
        return

    group_count = Counter(rel for rel, _ in all_files)
    print(f"Tìm thấy {len(all_files)} file trong {len(group_count)} bộ dataset:")
    for grp, cnt in sorted(group_count.items()):
        print(f"  {grp}: {cnt} file")

    runs = 0
    total_time = 0.0
    group_stats = {}
    current_group = None

    for rel_dir, filepath in all_files:
        basename = os.path.splitext(os.path.basename(filepath))[0]

        if rel_dir != current_group:
            current_group = rel_dir
            group_stats[rel_dir] = {
                "runs": 0,
                "time": 0.0,
                "sum_obj": 0,
                "sum_total": 0,
                "sum_greedy": 0,
                "improved": 0,
                "equal": 0,
                "worse": 0,
                "invalid": 0,
            }
            print(f"\n{'=' * 70}")
            print(f"BỘ: {rel_dir}")
            print(f"{'=' * 70}")

        result = solve_one_file(filepath, verbose=False)
        out_dir = os.path.join(RESULT_DIR, rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"{SOLVER_NAME}_{basename}.txt")
        write_result(out_file, result["assignments"], result["obj"], result["time"], result["status"])

        diff = result["obj"] - result["greedy_baseline"]
        if diff > 0:
            mark = f"+{diff} BETTER"
            group_stats[rel_dir]["improved"] += 1
        elif diff == 0:
            mark = "= greedy"
            group_stats[rel_dir]["equal"] += 1
        else:
            mark = f"{diff} WORSE"  # gần như không xảy ra do fallback
            group_stats[rel_dir]["worse"] += 1

        if not result["status"].startswith("VALID"):
            group_stats[rel_dir]["invalid"] += 1

        rate = result["obj"] / result["total_cc"] * 100 if result["total_cc"] else 0
        print(
            f"[{basename}] {result['obj']}/{result['total_cc']} ({rate:.1f}%) | "
            f"greedy={result['greedy_baseline']} | {mark} | {result['time']:.2f}s | {result['status']}"
        )

        runs += 1
        total_time += result["time"]
        st = group_stats[rel_dir]
        st["runs"] += 1
        st["time"] += result["time"]
        st["sum_obj"] += result["obj"]
        st["sum_total"] += result["total_cc"]
        st["sum_greedy"] += result["greedy_baseline"]

        # Ghi incremental overall
        write_overall(group_stats, runs, total_time)

    write_overall(group_stats, runs, total_time)
    print(f"\nHoàn thành {runs} file. Kết quả lưu tại: {RESULT_DIR}")


def write_overall(group_stats, runs, total_time):
    overall_file = os.path.join(RESULT_DIR, "Overall_Evaluation.txt")
    with open(overall_file, "w", encoding="utf-8") as f:
        f.write(f"Thuật toán: {SOLVER_NAME}\n")
        f.write(f"Tổng file: {runs}\n")
        f.write(f"Thời gian TB: {total_time / runs if runs else 0:.6f} giây\n\n")
        f.write(
            f"{'Bộ dataset':<25} {'Files':>6} {'Obj/Total':>18} {'Greedy':>10} "
            f"{'Better':>8} {'Equal':>8} {'Worse':>8} {'Invalid':>8} {'AvgTime':>10}\n"
        )
        f.write("-" * 115 + "\n")
        for grp, st in sorted(group_stats.items()):
            avg_time = st["time"] / st["runs"] if st["runs"] else 0
            f.write(
                f"{grp:<25} {st['runs']:>6} "
                f"{st['sum_obj']:>8}/{st['sum_total']:<8} "
                f"{st['sum_greedy']:>10} "
                f"{st['improved']:>8} {st['equal']:>8} {st['worse']:>8} {st['invalid']:>8} "
                f"{avg_time:>10.2f}\n"
            )


if __name__ == "__main__":
    main()
