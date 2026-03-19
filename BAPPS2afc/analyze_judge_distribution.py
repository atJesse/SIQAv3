import argparse
import glob
import json
import os
from collections import Counter

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze BAPPS 2AFC judge npy value distribution")
    parser.add_argument("--bapps_root", type=str, default="/data/dataset/BAPPS/dataset/2afc/val")
    parser.add_argument("--out_dir", type=str, default="/data/SIQAv3/BAPPS2afc/judge_stats")
    return parser.parse_args()


def to_scalar(arr: np.ndarray) -> float:
    return float(np.squeeze(arr))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    subsets = sorted([d for d in os.listdir(args.bapps_root) if os.path.isdir(os.path.join(args.bapps_root, d))])
    if not subsets:
        raise RuntimeError(f"No subsets found under {args.bapps_root}")

    global_values = []
    global_counter = Counter()
    global_shape_counter = Counter()
    global_dtype_counter = Counter()

    per_subset = {}

    for subset in subsets:
        judge_dir = os.path.join(args.bapps_root, subset, "judge")
        npy_files = sorted(glob.glob(os.path.join(judge_dir, "*.npy")))
        values = []
        value_counter = Counter()
        shape_counter = Counter()
        dtype_counter = Counter()

        for path in npy_files:
            arr = np.load(path)
            shape_counter[str(tuple(arr.shape))] += 1
            dtype_counter[str(arr.dtype)] += 1
            value = to_scalar(arr)
            values.append(value)
            value_counter[f"{value:.6f}"] += 1

        values_np = np.asarray(values, dtype=np.float64)
        bins = [-1e-9, 0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
        hist, edges = np.histogram(values_np, bins=bins)
        bin_labels = [
            "==0",
            "(0,0.2]",
            "(0.2,0.4]",
            "(0.4,0.5]",
            "(0.5,0.6]",
            "(0.6,0.8]",
            "(0.8,1.0]",
        ]

        subset_payload = {
            "count": int(len(values)),
            "min": float(values_np.min()) if len(values) else None,
            "max": float(values_np.max()) if len(values) else None,
            "mean": float(values_np.mean()) if len(values) else None,
            "num_eq_0": int(np.sum(values_np == 0.0)),
            "num_eq_0_5": int(np.sum(values_np == 0.5)),
            "num_eq_1": int(np.sum(values_np == 1.0)),
            "num_gt_0_5": int(np.sum(values_np > 0.5)),
            "num_lt_0_5": int(np.sum(values_np < 0.5)),
            "num_in_0_5_1": int(np.sum((values_np > 0.5) & (values_np < 1.0))),
            "num_in_0_0_5": int(np.sum((values_np > 0.0) & (values_np < 0.5))),
            "hist_bins": {label: int(count) for label, count in zip(bin_labels, hist.tolist())},
            "top_values": value_counter.most_common(20),
            "shapes": dict(shape_counter),
            "dtypes": dict(dtype_counter),
        }
        per_subset[subset] = subset_payload

        global_values.extend(values)
        global_counter.update(value_counter)
        global_shape_counter.update(shape_counter)
        global_dtype_counter.update(dtype_counter)

    global_np = np.asarray(global_values, dtype=np.float64)
    global_payload = {
        "count": int(len(global_values)),
        "min": float(global_np.min()),
        "max": float(global_np.max()),
        "mean": float(global_np.mean()),
        "num_eq_0": int(np.sum(global_np == 0.0)),
        "num_eq_0_5": int(np.sum(global_np == 0.5)),
        "num_eq_1": int(np.sum(global_np == 1.0)),
        "num_gt_0_5": int(np.sum(global_np > 0.5)),
        "num_lt_0_5": int(np.sum(global_np < 0.5)),
        "num_in_0_5_1": int(np.sum((global_np > 0.5) & (global_np < 1.0))),
        "num_in_0_0_5": int(np.sum((global_np > 0.0) & (global_np < 0.5))),
        "top_values": global_counter.most_common(30),
        "shapes": dict(global_shape_counter),
        "dtypes": dict(global_dtype_counter),
    }

    out_json = os.path.join(args.out_dir, "judge_distribution_summary.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"global": global_payload, "per_subset": per_subset}, f, indent=2)

    out_txt = os.path.join(args.out_dir, "judge_distribution_report.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"BAPPS root: {args.bapps_root}\n")
        f.write("\n[Global]\n")
        for k, v in global_payload.items():
            if k in {"top_values", "shapes", "dtypes"}:
                continue
            f.write(f"{k}: {v}\n")
        f.write(f"shapes: {global_payload['shapes']}\n")
        f.write(f"dtypes: {global_payload['dtypes']}\n")
        f.write(f"top_values: {global_payload['top_values'][:10]}\n")

        f.write("\n[Per Subset]\n")
        for subset, payload in per_subset.items():
            f.write(f"\n- {subset}\n")
            f.write(f"  count={payload['count']} mean={payload['mean']} min={payload['min']} max={payload['max']}\n")
            f.write(
                "  eq0={eq0} eq0.5={eq05} eq1={eq1} gt0.5={gt} lt0.5={lt}\n".format(
                    eq0=payload["num_eq_0"],
                    eq05=payload["num_eq_0_5"],
                    eq1=payload["num_eq_1"],
                    gt=payload["num_gt_0_5"],
                    lt=payload["num_lt_0_5"],
                )
            )

    print(f"Saved: {out_json}")
    print(f"Saved: {out_txt}")
    print(f"Global count={global_payload['count']} eq_0.5={global_payload['num_eq_0_5']}")


if __name__ == "__main__":
    main()
