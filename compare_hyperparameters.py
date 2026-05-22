"""
Compare 8 hyperparameter runs of evolutionary_timetabling_solver_cli.py.

Cach dung nhanh:
    python compare_hyperparameters_fixed.py --result-root Result
hoac neu thu muc cua ban la Results:
    python compare_hyperparameters_fixed.py --result-root Results

Script nay KHONG phu thuoc vao viec hard-code dung ten folder 100%.
No se tu quet cac folder con co file Overall_Evaluation.txt, doc Run_Config.txt
neu co, roi sinh bang + hinh so sanh.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


# Neu ban chi muon lay dung 8 bo thi de EXPECTED_TAGS nhu duoi.
# Script se uu tien sap xep theo thu tu nay neu tim thay tag/config tuong ung.
EXPECTED_ORDER = [
    "autoPop",
    "small_pop20",
    "low_crossover_0p6",
    "low_mutation_0p4",
    "high_kick_0p55",
    "high_mutation_0p9",
    "high_crossover_0p95",
    "large_pop60",
]

TAG_TO_LABEL = {
    "autoPop": "autoPop",
    "small_pop20": "pop20",
    "low_crossover_0p6": "low_cx",
    "low_mutation_0p4": "low_mut",
    "high_kick_0p55": "high_kick",
    "high_mutation_0p9": "high_mut",
    "high_crossover_0p95": "high_cx",
    "large_pop60": "pop60",
}

DATASET_ORDER = ["Adversarial", "Exponential", "Gaussian", "hu_stack", "hustack", "Poisson", "Uniform"]


# ---------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------

def parse_run_config(run_config_path: Path) -> Dict[str, str]:
    """Doc Run_Config.txt do solver ghi ra: moi dong dang key: value."""
    cfg: Dict[str, str] = {}
    if not run_config_path.exists():
        return cfg

    for raw in run_config_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if ":" not in raw:
            continue
        key, value = raw.split(":", 1)
        cfg[key.strip()] = value.strip()
    return cfg


def label_from_folder(folder: Path, cfg: Dict[str, str]) -> str:
    """Tao ten ngan gon cho config de hien thi tren bieu do."""
    tag = cfg.get("tag", "").strip()
    if tag and tag in TAG_TO_LABEL:
        return TAG_TO_LABEL[tag]
    if tag:
        return tag

    name = folder.name

    # Fallback theo ten folder neu khong co Run_Config.txt.
    for known_tag, label in TAG_TO_LABEL.items():
        if known_tag in name:
            return label

    # Auto run trong make_run_name cua solver thuong co popauto_eliteauto.
    if "popauto" in name and "eliteauto" in name:
        return "autoPop"

    # Cac fallback pho bien.
    pop = re.search(r"pop([^_]+)", name)
    cx = re.search(r"cx([^_]+)", name)
    mut = re.search(r"mut([^_]+)", name)
    kick = re.search(r"kick([^_]+)", name)
    pieces = []
    if pop:
        pieces.append(f"pop{pop.group(1)}")
    if cx:
        pieces.append(f"cx{cx.group(1)}")
    if mut:
        pieces.append(f"mut{mut.group(1)}")
    if kick:
        pieces.append(f"kick{kick.group(1)}")
    return "_".join(pieces) if pieces else name


def parse_overall_evaluation(file_path: Path) -> List[Dict[str, object]]:
    """
    Parse Overall_Evaluation.txt.

    Dong du lieu solver tao ra co dang:
    Adversarial                    5      123/200            120        3        2        0        0      20.00
    """
    rows: List[Dict[str, object]] = []
    if not file_path.exists():
        return rows

    lines = file_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    in_table = False

    pattern = re.compile(
        r"^\s*(?P<dataset>\S+)\s+"
        r"(?P<files>\d+)\s+"
        r"(?P<obj>\d+)\s*/\s*(?P<total>\d+)\s+"
        r"(?P<greedy>\d+)\s+"
        r"(?P<better>\d+)\s+"
        r"(?P<equal>\d+)\s+"
        r"(?P<worse>\d+)\s+"
        r"(?P<invalid>\d+)\s+"
        r"(?P<avg_time>[0-9.]+)\s*$"
    )

    for line in lines:
        if set(line.strip()) == {"-"}:
            in_table = True
            continue
        if not in_table or not line.strip():
            continue

        m = pattern.match(line)
        if not m:
            continue

        d = m.groupdict()
        obj = int(d["obj"])
        total = int(d["total"])
        greedy = int(d["greedy"])

        rows.append(
            {
                "Dataset": d["dataset"],
                "Files": int(d["files"]),
                "Obj": obj,
                "Total": total,
                "Greedy": greedy,
                "Better": int(d["better"]),
                "Equal": int(d["equal"]),
                "Worse": int(d["worse"]),
                "Invalid": int(d["invalid"]),
                "AvgTime": float(d["avg_time"]),
                "ObjRatio": obj / total if total else 0.0,
                "GreedyRatio": greedy / total if total else 0.0,
                "Gain": obj - greedy,
                "GainRatio": (obj - greedy) / total if total else 0.0,
            }
        )

    return rows


def discover_result_folders(result_root: Path, only_expected: bool = False) -> List[Path]:
    """Tim cac folder con co Overall_Evaluation.txt."""
    if not result_root.exists():
        raise FileNotFoundError(f"Khong ton tai thu muc result_root: {result_root}")

    folders = [p for p in result_root.iterdir() if p.is_dir() and (p / "Overall_Evaluation.txt").exists()]

    if only_expected:
        filtered = []
        for p in folders:
            cfg = parse_run_config(p / "Run_Config.txt")
            tag = cfg.get("tag", "")
            if tag in EXPECTED_ORDER or any(t in p.name for t in EXPECTED_ORDER) or ("popauto" in p.name and "eliteauto" in p.name):
                filtered.append(p)
        folders = filtered

    return sorted(folders, key=lambda p: folder_sort_key(p))


def folder_sort_key(folder: Path) -> Tuple[int, str]:
    cfg = parse_run_config(folder / "Run_Config.txt")
    tag = cfg.get("tag", "")
    label = label_from_folder(folder, cfg)
    candidates = [tag, label, folder.name]
    for i, expected in enumerate(EXPECTED_ORDER):
        if any(expected in c for c in candidates):
            return (i, folder.name)
    if label == "autoPop" or ("popauto" in folder.name and "eliteauto" in folder.name):
        return (0, folder.name)
    return (999, folder.name)


def collect_all_results(result_root: Path, only_expected: bool = False) -> pd.DataFrame:
    all_rows: List[Dict[str, object]] = []
    folders = discover_result_folders(result_root, only_expected=only_expected)

    if not folders:
        return pd.DataFrame()

    print(f"Tim thay {len(folders)} folder ket qua trong: {result_root}")

    used_labels: Dict[str, int] = {}
    for folder in folders:
        cfg = parse_run_config(folder / "Run_Config.txt")
        label = label_from_folder(folder, cfg)

        # Neu trung label do chay lai nhieu lan, them hau to de khong bi de len nhau.
        used_labels[label] = used_labels.get(label, 0) + 1
        display_label = label if used_labels[label] == 1 else f"{label}_{used_labels[label]}"

        overall = folder / "Overall_Evaluation.txt"
        rows = parse_overall_evaluation(overall)
        print(f"  - {display_label:<14} <- {folder.name}: {len(rows)} dataset")

        for row in rows:
            row.update(
                {
                    "Config": display_label,
                    "Folder": folder.name,
                    "Tag": cfg.get("tag", ""),
                    "PopSize": cfg.get("pop_size", ""),
                    "Elite": cfg.get("elite", ""),
                    "Crossover": cfg.get("crossover_rate", ""),
                    "Mutation": cfg.get("mutation_rate", ""),
                    "Kick": cfg.get("local_kick_rate", ""),
                    "TournamentK": cfg.get("tournament_k", ""),
                    "Seed": cfg.get("seed", ""),
                }
            )
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    if df.empty:
        return df

    # Sap xep dataset/config de hinh on dinh.
    config_order = [label_from_folder(p, parse_run_config(p / "Run_Config.txt")) for p in folders]
    # Xu ly label trung da gan _2.
    config_order = list(dict.fromkeys(df["Config"].tolist()))
    dataset_order = [d for d in DATASET_ORDER if d in set(df["Dataset"])] + sorted(set(df["Dataset"]) - set(DATASET_ORDER))

    df["Config"] = pd.Categorical(df["Config"], categories=config_order, ordered=True)
    df["Dataset"] = pd.Categorical(df["Dataset"], categories=dataset_order, ordered=True)
    return df.sort_values(["Dataset", "Config"]).reset_index(drop=True)


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def annotate_bars(ax, fmt: str = "{:.3f}", fontsize: int = 8) -> None:
    for patch in ax.patches:
        height = patch.get_height()
        if pd.isna(height) or height == 0:
            continue
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=0,
        )


def plot_grouped_metric(df: pd.DataFrame, out_dir: Path, metric: str, ylabel: str, title: str, filename: str, value_fmt: str) -> None:
    pivot = df.pivot_table(index="Dataset", columns="Config", values=metric, aggfunc="mean", observed=False)
    ax = pivot.plot(kind="bar", figsize=(16, 8), width=0.82)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="Config", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    if metric in {"ObjRatio", "GreedyRatio"}:
        ymin = max(0, min(df[metric].min() - 0.02, 0.9))
        ymax = min(1.02, max(df[metric].max() + 0.02, 1.0))
        ax.set_ylim(ymin, ymax)

    plt.tight_layout()
    path = out_dir / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Luu: {path}")


def plot_heatmap(df: pd.DataFrame, out_dir: Path, metric: str, title: str, filename: str, fmt: str = ".4f") -> None:
    pivot = df.pivot_table(index="Config", columns="Dataset", values=metric, aggfunc="mean", observed=False)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(pivot.values, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(title, fontsize=15, fontweight="bold")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=9)

    plt.tight_layout()
    path = out_dir / filename
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Luu: {path}")


def plot_better_equal_worse(df: pd.DataFrame, out_dir: Path) -> None:
    datasets = list(df["Dataset"].cat.categories if hasattr(df["Dataset"], "cat") else sorted(df["Dataset"].unique()))
    datasets = [d for d in datasets if d in set(df["Dataset"].astype(str))]

    n = len(datasets)
    rows = 2 if n > 3 else 1
    cols = 3 if n > 3 else n
    fig, axes = plt.subplots(rows, cols, figsize=(18, 10 if rows == 2 else 5), squeeze=False)
    axes_flat = axes.flatten()

    fig.suptitle("Better / Equal / Worse so voi Greedy theo tung dataset", fontsize=16, fontweight="bold")

    for idx, dataset in enumerate(datasets):
        ax = axes_flat[idx]
        sub = df[df["Dataset"].astype(str) == str(dataset)].copy()
        x = range(len(sub))
        width = 0.25

        ax.bar([i - width for i in x], sub["Better"], width, label="Better")
        ax.bar(list(x), sub["Equal"], width, label="Equal")
        ax.bar([i + width for i in x], sub["Worse"], width, label="Worse")
        ax.set_title(str(dataset), fontweight="bold")
        ax.set_ylabel("So file")
        ax.set_xticks(list(x))
        ax.set_xticklabels(sub["Config"].astype(str), rotation=45, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

        for patch in ax.patches:
            h = patch.get_height()
            if h > 0:
                ax.text(patch.get_x() + patch.get_width() / 2, h, f"{int(h)}", ha="center", va="bottom", fontsize=8)

    for j in range(idx + 1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    plt.tight_layout()
    path = out_dir / "hyperparameter_better_equal_worse.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Luu: {path}")


def plot_total_gain(df: pd.DataFrame, out_dir: Path) -> None:
    summary = df.groupby("Config", observed=False).agg(TotalGain=("Gain", "sum"), AvgGainRatio=("GainRatio", "mean")).reset_index()
    summary = summary.sort_values("TotalGain", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(summary["Config"].astype(str), summary["TotalGain"])
    ax.set_title("Tong muc cai thien so voi Greedy theo config", fontsize=15, fontweight="bold")
    ax.set_xlabel("Config")
    ax.set_ylabel("Tong Gain = sum(Obj - Greedy)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    annotate_bars(ax, fmt="{:.0f}", fontsize=9)

    plt.tight_layout()
    path = out_dir / "hyperparameter_total_gain.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Luu: {path}")


def generate_outputs(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    ensure_dir(out_dir)

    detail_path = out_dir / "hyperparameter_comparison_detail.csv"
    df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"✓ Luu: {detail_path}")

    summary = (
        df.groupby("Config", observed=False)
        .agg(
            Datasets=("Dataset", "nunique"),
            Files=("Files", "sum"),
            Obj=("Obj", "sum"),
            Total=("Total", "sum"),
            Greedy=("Greedy", "sum"),
            Better=("Better", "sum"),
            Equal=("Equal", "sum"),
            Worse=("Worse", "sum"),
            Invalid=("Invalid", "sum"),
            AvgTimeMean=("AvgTime", "mean"),
        )
        .reset_index()
    )
    summary["ObjRatio"] = summary["Obj"] / summary["Total"]
    summary["GreedyRatio"] = summary["Greedy"] / summary["Total"]
    summary["Gain"] = summary["Obj"] - summary["Greedy"]
    summary["GainRatio"] = summary["Gain"] / summary["Total"]
    summary = summary.sort_values(["ObjRatio", "Gain", "AvgTimeMean"], ascending=[False, False, True])

    summary_path = out_dir / "hyperparameter_comparison_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"✓ Luu: {summary_path}")

    # Hinh chinh nen dua vao bao cao.
    plot_grouped_metric(
        df,
        out_dir,
        metric="ObjRatio",
        ylabel="Obj / Total",
        title="So sanh ty le xep duoc Obj/Total giua cac bo hyperparameter",
        filename="hyperparameter_obj_ratio_by_dataset.png",
        value_fmt="{:.4f}",
    )
    plot_grouped_metric(
        df,
        out_dir,
        metric="GainRatio",
        ylabel="(Obj - Greedy) / Total",
        title="Muc cai thien so voi Greedy theo dataset",
        filename="hyperparameter_gain_ratio_by_dataset.png",
        value_fmt="{:.4f}",
    )
    plot_better_equal_worse(df, out_dir)
    plot_grouped_metric(
        df,
        out_dir,
        metric="AvgTime",
        ylabel="Avg time / file (seconds)",
        title="So sanh thoi gian chay trung binh theo dataset",
        filename="hyperparameter_avg_time_by_dataset.png",
        value_fmt="{:.1f}",
    )
    plot_heatmap(
        df,
        out_dir,
        metric="ObjRatio",
        title="Heatmap Obj/Total: Config vs Dataset",
        filename="hyperparameter_obj_ratio_heatmap.png",
        fmt=".4f",
    )
    plot_heatmap(
        df,
        out_dir,
        metric="GainRatio",
        title="Heatmap GainRatio: Config vs Dataset",
        filename="hyperparameter_gain_ratio_heatmap.png",
        fmt=".4f",
    )
    plot_total_gain(df, out_dir)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare hyperparameter results from evolutionary timetabling runs.")
    parser.add_argument("--result-root", default="Result", help="Thu muc chua cac folder ket qua, vi du Result hoac Results")
    parser.add_argument("--out-dir", default=None, help="Thu muc luu CSV/hinh. Mac dinh: <result-root>/Hyperparameter_Comparison")
    parser.add_argument("--only-expected", action="store_true", help="Chi lay cac folder khop 8 tag hyperparameter du kien")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    out_dir = Path(args.out_dir) if args.out_dir else result_root / "Hyperparameter_Comparison"

    print("Bat dau compare hyperparameters...")
    print(f"Result root: {result_root}")

    df = collect_all_results(result_root, only_expected=args.only_expected)
    if df.empty:
        print("\n❌ Khong doc duoc du lieu.")
        print("Kiem tra lai:")
        print("  1) Dung --result-root Result hay --result-root Results")
        print("  2) Moi folder run phai co Overall_Evaluation.txt")
        print("  3) Nen chay file evo batch xong roi moi compare")
        return

    summary = generate_outputs(df, out_dir)

    print("\n" + "=" * 120)
    print("BANG TOM TAT XEP HANG CONFIG")
    print("=" * 120)
    cols = ["Config", "Files", "Obj", "Total", "ObjRatio", "Greedy", "Gain", "GainRatio", "Better", "Equal", "Worse", "AvgTimeMean"]
    print(summary[cols].to_string(index=False))
    print("=" * 120)
    print(f"\n✅ Hoan thanh. Ket qua luu trong: {out_dir}")


if __name__ == "__main__":
    main()
