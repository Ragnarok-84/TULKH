"""
Plot AutoPop vs Greedy baseline for Evolutionary Timetabling report.

Script này dùng kết quả đã lưu trong cây thư mục:
    Result/
      Evolutionary_Timetabling_autoPop_cx0p85_mut0p75_kick0p35_tk3_seed42/
        Overall_Evaluation.txt
      Greedy_Heuristic/                         # optional
        Overall_Evaluation.txt                  # optional

Lưu ý quan trọng:
- File Evolutionary đã ghi sẵn cột Greedy trong Overall_Evaluation.txt.
- Vì vậy script ưu tiên dùng chính cột Greedy của AutoPop để so sánh công bằng.
- Nếu muốn ép đọc folder Greedy_Heuristic thì dùng --prefer-greedy-folder.

Chạy:
    python plot_autopop_vs_greedy.py
hoặc:
    python plot_autopop_vs_greedy.py --result-root Result
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt


# ==============================
# Helpers
# ==============================


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def parse_overall_evaluation(file_path: Path) -> pd.DataFrame:
    """Parse Overall_Evaluation.txt thành DataFrame theo từng bộ dataset."""
    rows: List[Dict[str, object]] = []

    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    with file_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    in_table = False
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if set(line) == {"-"} or re.fullmatch(r"-+", line):
            in_table = True
            continue

        if not in_table:
            continue

        parts = line.split()
        if len(parts) < 9:
            continue

        # Format từ evolutionary_timetabling_solver_cli.py:
        # Dataset Files Obj/Total Greedy Better Equal Worse Invalid AvgTime
        dataset = parts[0]
        try:
            files = int(parts[1])
            obj_str, total_str = parts[2].split("/")
            obj = int(obj_str)
            total = int(total_str)
            greedy = int(parts[3])
            better = int(parts[4])
            equal = int(parts[5])
            worse = int(parts[6])
            invalid = int(parts[7])
            avg_time = float(parts[8])
        except Exception:
            continue

        rows.append(
            {
                "Dataset": dataset,
                "Files": files,
                "Obj": obj,
                "Total": total,
                "Greedy": greedy,
                "Better": better,
                "Equal": equal,
                "Worse": worse,
                "Invalid": invalid,
                "AvgTime": avg_time,
                "ObjRatio": safe_div(obj, total),
                "GreedyRatio": safe_div(greedy, total),
                "Gain": obj - greedy,
                "GainRatio": safe_div(obj - greedy, total),
            }
        )

    if not rows:
        raise ValueError(f"Không parse được bảng dữ liệu trong: {file_path}")

    return pd.DataFrame(rows)


def find_autopop_folder(result_root: Path, explicit_name: Optional[str] = None) -> Path:
    """Tự tìm folder AutoPop trong Result."""
    if explicit_name:
        p = result_root / explicit_name
        if not p.exists():
            raise FileNotFoundError(f"Không thấy autoPop folder được chỉ định: {p}")
        return p

    candidates = []
    for p in result_root.iterdir():
        if not p.is_dir():
            continue
        name_lower = p.name.lower()
        if "autopop" in name_lower or "popauto" in name_lower:
            if (p / "Overall_Evaluation.txt").exists():
                candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            "Không tìm thấy folder AutoPop. Hãy truyền rõ bằng --autopop-folder <ten_folder>."
        )

    # Nếu có nhiều folder, ưu tiên folder có chữ autopop trong tag.
    candidates.sort(key=lambda x: ("autopop" not in x.name.lower(), x.name))
    return candidates[0]


def find_greedy_folder(result_root: Path, explicit_name: Optional[str] = None) -> Optional[Path]:
    if explicit_name:
        p = result_root / explicit_name
        return p if p.exists() else None

    for name in ["Greedy_Heuristic", "Greedy", "greedy", "GreedyHeuristic"]:
        p = result_root / name
        if (p / "Overall_Evaluation.txt").exists():
            return p
    return None


def load_compare_dataframe(
    result_root: Path,
    autopop_folder_name: Optional[str],
    greedy_folder_name: Optional[str],
    prefer_greedy_folder: bool,
) -> pd.DataFrame:
    autopop_folder = find_autopop_folder(result_root, autopop_folder_name)
    auto_df = parse_overall_evaluation(autopop_folder / "Overall_Evaluation.txt")

    auto_df = auto_df.sort_values("Dataset").reset_index(drop=True)
    auto_df["AutoPop"] = auto_df["Obj"]
    auto_df["AutoPopRatio"] = auto_df["ObjRatio"]

    greedy_folder = find_greedy_folder(result_root, greedy_folder_name)

    # Mặc định: dùng cột Greedy trong file AutoPop, vì đây là baseline cùng lúc chạy với AutoPop.
    auto_df["GreedySource"] = "Greedy column inside AutoPop Overall_Evaluation.txt"

    if prefer_greedy_folder and greedy_folder is not None:
        try:
            greedy_df = parse_overall_evaluation(greedy_folder / "Overall_Evaluation.txt")
            greedy_df = greedy_df[["Dataset", "Obj", "Total"]].rename(
                columns={"Obj": "GreedyFromFolder", "Total": "GreedyTotalFromFolder"}
            )
            auto_df = auto_df.merge(greedy_df, on="Dataset", how="left")
            auto_df["Greedy"] = auto_df["GreedyFromFolder"].fillna(auto_df["Greedy"]).astype(int)
            auto_df["GreedySource"] = f"{greedy_folder.name}/Overall_Evaluation.txt"
        except Exception as e:
            print(f"⚠ Không đọc được Greedy folder, dùng cột Greedy trong AutoPop. Lỗi: {e}")

    auto_df["GreedyRatio"] = auto_df.apply(lambda r: safe_div(r["Greedy"], r["Total"]), axis=1)
    auto_df["Gain"] = auto_df["AutoPop"] - auto_df["Greedy"]
    auto_df["GainRatio"] = auto_df.apply(lambda r: safe_div(r["Gain"], r["Total"]), axis=1)

    print(f"AutoPop folder: {autopop_folder}")
    if greedy_folder:
        print(f"Greedy folder tìm thấy: {greedy_folder}")
    else:
        print("Greedy folder: không thấy, dùng cột Greedy trong AutoPop.")

    return auto_df


# ==============================
# Plot functions
# ==============================


def annotate_bars(ax, bars, fmt="{:.4f}", fontsize=8, rotation=0):
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=rotation,
        )


def plot_ratio_by_dataset(df: pd.DataFrame, out_dir: Path):
    datasets = df["Dataset"].tolist()
    x = range(len(datasets))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_g = ax.bar([i - width / 2 for i in x], df["GreedyRatio"], width, label="Greedy")
    bars_a = ax.bar([i + width / 2 for i in x], df["AutoPopRatio"], width, label="AutoPop Evolutionary")

    ax.set_title("So sánh tỉ lệ xếp lịch thành công: AutoPop Evolutionary vs Greedy", fontweight="bold")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Objective Ratio = Obj / Total")
    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_ylim(max(0, min(df["GreedyRatio"].min(), df["AutoPopRatio"].min()) - 0.01), 1.005)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    annotate_bars(ax, bars_g, "{:.4f}")
    annotate_bars(ax, bars_a, "{:.4f}")

    fig.tight_layout()
    fig.savefig(out_dir / "autopop_vs_greedy_obj_ratio_by_dataset.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_objective_count_by_dataset(df: pd.DataFrame, out_dir: Path):
    datasets = df["Dataset"].tolist()
    x = range(len(datasets))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 6))
    bars_g = ax.bar([i - width / 2 for i in x], df["Greedy"], width, label="Greedy")
    bars_a = ax.bar([i + width / 2 for i in x], df["AutoPop"], width, label="AutoPop Evolutionary")

    ax.set_title("So sánh tổng số lớp-môn được xếp: AutoPop Evolutionary vs Greedy", fontweight="bold")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Objective Value")
    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    annotate_bars(ax, bars_g, "{:.0f}", fontsize=8, rotation=45)
    annotate_bars(ax, bars_a, "{:.0f}", fontsize=8, rotation=45)

    fig.tight_layout()
    fig.savefig(out_dir / "autopop_vs_greedy_objective_count_by_dataset.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gain_by_dataset(df: pd.DataFrame, out_dir: Path):
    data = df.sort_values("Gain", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(data["Dataset"], data["Gain"])
    ax.axhline(0, linewidth=1)
    ax.set_title("Mức cải thiện của AutoPop so với Greedy theo Dataset", fontweight="bold")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Gain = AutoPop Obj - Greedy Obj")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    annotate_bars(ax, bars, "{:.0f}", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "autopop_vs_greedy_gain_by_dataset.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_gain_ratio_by_dataset(df: pd.DataFrame, out_dir: Path):
    data = df.sort_values("GainRatio", ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(data["Dataset"], data["GainRatio"] * 100.0)
    ax.axhline(0, linewidth=1)
    ax.set_title("GainRatio của AutoPop so với Greedy theo Dataset", fontweight="bold")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("GainRatio (%)")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    annotate_bars(ax, bars, "{:.3f}%", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_dir / "autopop_vs_greedy_gain_ratio_by_dataset.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_overall_pie_or_bar(df: pd.DataFrame, out_dir: Path):
    total_auto = int(df["AutoPop"].sum())
    total_greedy = int(df["Greedy"].sum())
    total_all = int(df["Total"].sum())
    gain = total_auto - total_greedy

    labels = ["Greedy", "AutoPop Evolutionary"]
    values = [total_greedy, total_auto]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, values)
    ax.set_title("Tổng objective trên toàn bộ 60 instances", fontweight="bold")
    ax.set_ylabel("Objective Value")
    ax.grid(axis="y", alpha=0.25)
    annotate_bars(ax, bars, "{:.0f}", fontsize=10)

    summary_text = (
        f"Total capacity: {total_all:,}\n"
        f"Greedy: {total_greedy:,} ({safe_div(total_greedy, total_all):.4%})\n"
        f"AutoPop: {total_auto:,} ({safe_div(total_auto, total_all):.4%})\n"
        f"Gain: +{gain:,} ({safe_div(gain, total_all):.4%})"
    )
    ax.text(0.5, 0.05, summary_text, transform=ax.transAxes, ha="center", va="bottom",
            bbox=dict(boxstyle="round", alpha=0.1))

    fig.tight_layout()
    fig.savefig(out_dir / "autopop_vs_greedy_overall_objective.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_better_equal_worse(df: pd.DataFrame, out_dir: Path):
    datasets = df["Dataset"].tolist()
    x = range(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar([i - width for i in x], df["Better"], width, label="Better than Greedy")
    b2 = ax.bar(list(x), df["Equal"], width, label="Equal to Greedy")
    b3 = ax.bar([i + width for i in x], df["Worse"], width, label="Worse than Greedy")

    ax.set_title("Số instance AutoPop tốt hơn / bằng / kém hơn Greedy", fontweight="bold")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Số file")
    ax.set_xticks(list(x))
    ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    for bars in [b1, b2, b3]:
        annotate_bars(ax, bars, "{:.0f}", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "autopop_vs_greedy_better_equal_worse.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_summary(df: pd.DataFrame, out_dir: Path):
    summary_cols = [
        "Dataset", "Files", "Total", "Greedy", "AutoPop", "Gain",
        "GreedyRatio", "AutoPopRatio", "GainRatio", "Better", "Equal", "Worse", "Invalid", "AvgTime",
    ]
    detail = df[summary_cols].copy()
    detail.to_csv(out_dir / "autopop_vs_greedy_by_dataset.csv", index=False, encoding="utf-8-sig")

    total_row = {
        "Dataset": "ALL",
        "Files": int(df["Files"].sum()),
        "Total": int(df["Total"].sum()),
        "Greedy": int(df["Greedy"].sum()),
        "AutoPop": int(df["AutoPop"].sum()),
        "Gain": int(df["Gain"].sum()),
        "GreedyRatio": safe_div(df["Greedy"].sum(), df["Total"].sum()),
        "AutoPopRatio": safe_div(df["AutoPop"].sum(), df["Total"].sum()),
        "GainRatio": safe_div(df["Gain"].sum(), df["Total"].sum()),
        "Better": int(df["Better"].sum()),
        "Equal": int(df["Equal"].sum()),
        "Worse": int(df["Worse"].sum()),
        "Invalid": int(df["Invalid"].sum()),
        "AvgTime": float(df["AvgTime"].mean()),
    }
    summary = pd.DataFrame([total_row])
    summary.to_csv(out_dir / "autopop_vs_greedy_summary.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 120)
    print("BẢNG SO SÁNH AUTOPOP EVOLUTIONARY VS GREEDY")
    print("=" * 120)
    print(detail.to_string(index=False))
    print("-" * 120)
    print(summary.to_string(index=False))
    print("=" * 120)


def main():
    parser = argparse.ArgumentParser(description="Plot AutoPop Evolutionary vs Greedy baseline")
    parser.add_argument("--result-root", default="Result", help="Thư mục gốc chứa kết quả, ví dụ Result hoặc Results")
    parser.add_argument("--autopop-folder", default=None, help="Tên folder AutoPop nếu muốn chỉ định rõ")
    parser.add_argument("--greedy-folder", default=None, help="Tên folder Greedy nếu muốn chỉ định rõ")
    parser.add_argument("--prefer-greedy-folder", action="store_true", help="Ưu tiên đọc Greedy từ folder Greedy_Heuristic thay vì cột Greedy trong AutoPop")
    parser.add_argument("--out-dir", default=None, help="Folder lưu hình. Mặc định: Result/AutoPop_vs_Greedy")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    if not result_root.exists():
        raise FileNotFoundError(f"Không thấy thư mục result-root: {result_root}")

    out_dir = Path(args.out_dir) if args.out_dir else result_root / "AutoPop_vs_Greedy"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_compare_dataframe(
        result_root=result_root,
        autopop_folder_name=args.autopop_folder,
        greedy_folder_name=args.greedy_folder,
        prefer_greedy_folder=args.prefer_greedy_folder,
    )

    make_summary(df, out_dir)
    plot_ratio_by_dataset(df, out_dir)
    plot_objective_count_by_dataset(df, out_dir)
    plot_gain_by_dataset(df, out_dir)
    plot_gain_ratio_by_dataset(df, out_dir)
    plot_overall_pie_or_bar(df, out_dir)
    plot_better_equal_worse(df, out_dir)

    print(f"\n✅ Hoàn thành. Kết quả lưu trong: {out_dir}")
    print("Các hình chính nên đưa vào báo cáo:")
    print(f"  - {out_dir / 'autopop_vs_greedy_obj_ratio_by_dataset.png'}")
    print(f"  - {out_dir / 'autopop_vs_greedy_gain_by_dataset.png'}")
    print(f"  - {out_dir / 'autopop_vs_greedy_overall_objective.png'}")


if __name__ == "__main__":
    main()
