#!/usr/bin/env python3
from __future__ import annotations

import ast
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


WORKSPACE = Path(__file__).resolve().parent.parent
DATA_DIR = WORKSPACE / "data"
RESULTS_DIR = DATA_DIR / "results"
AMPLITUDES_DIR = DATA_DIR / "amplitudes"
OUTPUTS_DIR = WORKSPACE / "outputs"
IMAGES_DIR = WORKSPACE / "report" / "images"

OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

FILE_RE = re.compile(
    r"N(?P<N>\d+)_d(?P<d>\d+)_r(?P<r>\d+)_(?P<protocol>.+?)_(?P<kind>counts|ideal_bitstring|amplitudes)\.json$"
)


@dataclass
class XEBRecord:
    N: int
    d: int
    r: int
    total_shots: int
    matched_shots: int
    unique_bitstrings: int
    matched_unique_bitstrings: int
    mean_ideal_prob: float
    xeb_fidelity: float
    xeb_std_error: float
    xeb_sample_std: float
    min_ideal_prob: float
    max_ideal_prob: float
    counts_path: str
    amplitudes_path: str


@dataclass
class MBRecord:
    N: int
    d: int
    r: int
    total_shots: int
    unique_bitstrings: int
    ideal_success_count: int
    ideal_success_prob: float
    ideal_success_std_error: float
    wilson_low: float
    wilson_high: float
    counts_path: str
    ideal_path: str
    ideal_bitstring: str


@dataclass
class AggregateRecord:
    protocol: str
    metric: str
    N: int
    d: int
    num_instances: int
    mean: float
    std: float
    sem: float
    min_value: float
    max_value: float


@dataclass
class MissingPair:
    protocol: str
    N: int
    d: int
    r: int
    missing: str
    counts_path: str
    aux_path: str


def parse_metadata(path: Path) -> Optional[dict]:
    match = FILE_RE.search(path.name)
    if not match:
        return None
    meta = match.groupdict()
    return {
        "N": int(meta["N"]),
        "d": int(meta["d"]),
        "r": int(meta["r"]),
        "protocol": meta["protocol"],
        "kind": meta["kind"],
        "path": path,
    }


def normalize_bitstring_key(key) -> str:
    if isinstance(key, list):
        return "".join(str(int(x)) for x in key)
    if isinstance(key, tuple):
        return "".join(str(int(x)) for x in key)
    if isinstance(key, str):
        stripped = key.strip()
        if stripped.startswith("(") and stripped.endswith(")"):
            value = ast.literal_eval(stripped)
            if isinstance(value, tuple):
                return "".join(str(int(x)) for x in value)
        binary_chars = [ch for ch in stripped if ch in "01"]
        if binary_chars:
            return "".join(binary_chars)
    raise ValueError(f"Unsupported bitstring format: {key!r}")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_counts(path: Path) -> Dict[str, int]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Counts file is not a dict: {path}")
    counts = {}
    for key, value in raw.items():
        counts[normalize_bitstring_key(key)] = int(value)
    return counts


def parse_amplitude_or_probability(value) -> float:
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if numeric >= 0 else abs(numeric)
    if isinstance(value, str):
        text = value.strip()
        try:
            complex_val = complex(text)
            return abs(complex_val) ** 2
        except ValueError:
            numeric = float(text)
            return numeric if numeric >= 0 else abs(numeric)
    if isinstance(value, list) and len(value) == 2:
        complex_val = complex(value[0], value[1])
        return abs(complex_val) ** 2
    raise ValueError(f"Unsupported amplitude/probability format: {value!r}")


def load_ideal_probabilities(path: Path) -> Dict[str, float]:
    raw = load_json(path)
    if not isinstance(raw, dict):
        raise ValueError(f"Amplitude file is not a dict: {path}")
    return {normalize_bitstring_key(key): parse_amplitude_or_probability(value) for key, value in raw.items()}


def load_ideal_bitstring(path: Path) -> str:
    raw = load_json(path)
    return normalize_bitstring_key(raw)


def weighted_mean_and_se(values: List[float], weights: List[int]) -> Tuple[float, float, float]:
    total_weight = sum(weights)
    if total_weight <= 0:
        return float("nan"), float("nan"), float("nan")
    arr_v = np.asarray(values, dtype=float)
    arr_w = np.asarray(weights, dtype=float)
    mean_value = float(np.sum(arr_v * arr_w) / total_weight)
    if total_weight <= 1:
        return mean_value, float("nan"), 0.0
    variance = float(np.sum(arr_w * (arr_v - mean_value) ** 2) / (total_weight - 1))
    sample_std = math.sqrt(max(variance, 0.0))
    se = sample_std / math.sqrt(total_weight)
    return mean_value, se, sample_std


def wilson_interval(successes: int, total: int, z: float = 1.0) -> Tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total)))
    return max(0.0, center - margin), min(1.0, center + margin)


def collect_files(root: Path, suffix: str) -> Dict[Tuple[int, int, int, str], Path]:
    files = {}
    for path in sorted(root.rglob(suffix)):
        meta = parse_metadata(path)
        if not meta:
            continue
        key = (meta["N"], meta["d"], meta["r"], meta["protocol"])
        files[key] = path
    return files


def analyze_xeb() -> Tuple[List[XEBRecord], List[MissingPair]]:
    count_files = collect_files(RESULTS_DIR, "*_counts.json")
    amplitude_files = collect_files(AMPLITUDES_DIR, "*_amplitudes.json")
    records: List[XEBRecord] = []
    missing: List[MissingPair] = []

    xeb_count_keys = [key for key in count_files if key[3] == "XEB"]
    for key in sorted(xeb_count_keys):
        counts_path = count_files[key]
        amp_path = amplitude_files.get(key)
        N, d, r, protocol = key
        if amp_path is None:
            missing.append(
                MissingPair(
                    protocol=protocol,
                    N=N,
                    d=d,
                    r=r,
                    missing="amplitudes",
                    counts_path=str(counts_path.relative_to(WORKSPACE)),
                    aux_path="",
                )
            )
            continue

        counts = load_counts(counts_path)
        ideal_probs = load_ideal_probabilities(amp_path)
        matched = [(bitstring, count, ideal_probs[bitstring]) for bitstring, count in counts.items() if bitstring in ideal_probs]
        total_shots = int(sum(counts.values()))
        matched_shots = int(sum(count for _, count, _ in matched))
        if matched_shots <= 0:
            continue

        xeb_values = [(2 ** N) * prob - 1.0 for _, _, prob in matched]
        weights = [count for _, count, _ in matched]
        fidelity, se, sample_std = weighted_mean_and_se(xeb_values, weights)
        probs = [prob for _, _, prob in matched]

        records.append(
            XEBRecord(
                N=N,
                d=d,
                r=r,
                total_shots=total_shots,
                matched_shots=matched_shots,
                unique_bitstrings=len(counts),
                matched_unique_bitstrings=len(matched),
                mean_ideal_prob=float(np.average(np.asarray(probs), weights=np.asarray(weights, dtype=float))),
                xeb_fidelity=fidelity,
                xeb_std_error=se,
                xeb_sample_std=sample_std,
                min_ideal_prob=min(probs),
                max_ideal_prob=max(probs),
                counts_path=str(counts_path.relative_to(WORKSPACE)),
                amplitudes_path=str(amp_path.relative_to(WORKSPACE)),
            )
        )

    return records, missing


def analyze_mb() -> Tuple[List[MBRecord], List[MissingPair]]:
    count_files = collect_files(RESULTS_DIR, "*_counts.json")
    ideal_files = collect_files(RESULTS_DIR, "*_ideal_bitstring.json")
    records: List[MBRecord] = []
    missing: List[MissingPair] = []

    mb_count_keys = [key for key in count_files if key[3] == "MB"]
    for key in sorted(mb_count_keys):
        counts_path = count_files[key]
        ideal_path = ideal_files.get(key)
        N, d, r, protocol = key
        if ideal_path is None:
            missing.append(
                MissingPair(
                    protocol=protocol,
                    N=N,
                    d=d,
                    r=r,
                    missing="ideal_bitstring",
                    counts_path=str(counts_path.relative_to(WORKSPACE)),
                    aux_path="",
                )
            )
            continue

        counts = load_counts(counts_path)
        ideal_bitstring = load_ideal_bitstring(ideal_path)
        total_shots = int(sum(counts.values()))
        success_count = int(counts.get(ideal_bitstring, 0))
        success_prob = success_count / total_shots if total_shots else float("nan")
        se = math.sqrt(success_prob * (1.0 - success_prob) / total_shots) if total_shots else float("nan")
        wilson_low, wilson_high = wilson_interval(success_count, total_shots, z=1.0)

        records.append(
            MBRecord(
                N=N,
                d=d,
                r=r,
                total_shots=total_shots,
                unique_bitstrings=len(counts),
                ideal_success_count=success_count,
                ideal_success_prob=success_prob,
                ideal_success_std_error=se,
                wilson_low=wilson_low,
                wilson_high=wilson_high,
                counts_path=str(counts_path.relative_to(WORKSPACE)),
                ideal_path=str(ideal_path.relative_to(WORKSPACE)),
                ideal_bitstring=ideal_bitstring,
            )
        )

    return records, missing


def aggregate(records: Iterable, protocol: str, metric: str) -> List[AggregateRecord]:
    grouped: Dict[Tuple[int, int], List[float]] = defaultdict(list)
    for record in records:
        grouped[(record.N, record.d)].append(float(getattr(record, metric)))

    output = []
    for (N, d), values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        sem = float(std / math.sqrt(len(arr))) if len(arr) > 1 else 0.0
        output.append(
            AggregateRecord(
                protocol=protocol,
                metric=metric,
                N=N,
                d=d,
                num_instances=len(values),
                mean=float(arr.mean()),
                std=std,
                sem=sem,
                min_value=float(arr.min()),
                max_value=float(arr.max()),
            )
        )
    return output


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_xeb_depth(records: List[XEBRecord]) -> Optional[str]:
    subset = [r for r in records if r.N == 40]
    if not subset:
        return None
    grouped = defaultdict(list)
    for rec in subset:
        grouped[rec.d].append(rec.xeb_fidelity)
    depths = sorted(grouped)
    means = [float(np.mean(grouped[d])) for d in depths]
    sems = [float(np.std(grouped[d], ddof=1) / math.sqrt(len(grouped[d]))) if len(grouped[d]) > 1 else 0.0 for d in depths]

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(depths, means, yerr=sems, marker="o", capsize=3, linewidth=2, color="#1f77b4")
    plt.axhline(0.0, linestyle="--", color="black", linewidth=1)
    plt.xlabel("Circuit depth d")
    plt.ylabel("Mean linear XEB fidelity")
    plt.title("N=40 verification: XEB fidelity vs depth")
    plt.grid(alpha=0.25)
    out = IMAGES_DIR / "xeb_fidelity_vs_depth_N40.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return str(out.relative_to(WORKSPACE))


def plot_xeb_size(records: List[XEBRecord]) -> Optional[str]:
    subset = [r for r in records if r.d == 12]
    if not subset:
        return None
    grouped = defaultdict(list)
    for rec in subset:
        grouped[rec.N].append(rec.xeb_fidelity)
    Ns = sorted(grouped)
    means = [float(np.mean(grouped[N])) for N in Ns]
    sems = [float(np.std(grouped[N], ddof=1) / math.sqrt(len(grouped[N]))) if len(grouped[N]) > 1 else 0.0 for N in Ns]

    plt.figure(figsize=(7, 4.5))
    plt.errorbar(Ns, means, yerr=sems, marker="s", capsize=3, linewidth=2, color="#d62728")
    plt.axhline(0.0, linestyle="--", color="black", linewidth=1)
    plt.xlabel("Qubit count N")
    plt.ylabel("Mean linear XEB fidelity")
    plt.title("Depth d=12: XEB fidelity vs system size")
    plt.grid(alpha=0.25)
    out = IMAGES_DIR / "xeb_fidelity_vs_size_d12.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return str(out.relative_to(WORKSPACE))


def plot_mb_depth(records: List[MBRecord]) -> Optional[str]:
    if not records:
        return None
    grouped = defaultdict(lambda: defaultdict(list))
    for rec in records:
        grouped[rec.N][rec.d].append(rec.ideal_success_prob)

    plt.figure(figsize=(8, 5))
    for N in sorted(grouped):
        depths = sorted(grouped[N])
        means = [float(np.mean(grouped[N][d])) for d in depths]
        plt.plot(depths, means, marker="o", linewidth=1.7, label=f"N={N}")
    plt.xlabel("Circuit depth d")
    plt.ylabel("Mean probability of ideal MB bitstring")
    plt.title("MB verification signal across available depths")
    plt.grid(alpha=0.25)
    plt.legend(ncol=2, fontsize=9)
    out = IMAGES_DIR / "mb_success_vs_depth.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return str(out.relative_to(WORKSPACE))


def plot_data_overview(xeb_records: List[XEBRecord], mb_records: List[MBRecord], missing_xeb: List[MissingPair]) -> str:
    xeb_counter = Counter((r.N, r.d) for r in xeb_records)
    mb_counter = Counter((r.N, r.d) for r in mb_records)
    missing_counter = Counter((m.N, m.d) for m in missing_xeb)
    all_keys = sorted(set(xeb_counter) | set(mb_counter) | set(missing_counter))
    labels = [f"N{N}\nd{d}" for N, d in all_keys]
    x = np.arange(len(all_keys))
    width = 0.26

    plt.figure(figsize=(max(10, len(all_keys) * 0.55), 4.8))
    plt.bar(x - width, [xeb_counter[k] for k in all_keys], width=width, label="XEB analyzed")
    plt.bar(x, [mb_counter[k] for k in all_keys], width=width, label="MB analyzed")
    plt.bar(x + width, [missing_counter[k] for k in all_keys], width=width, label="XEB missing amplitudes")
    plt.xticks(x, labels, rotation=90)
    plt.ylabel("Number of circuit instances")
    plt.title("Data coverage by (N, d)")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    out = IMAGES_DIR / "data_overview_by_setting.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return str(out.relative_to(WORKSPACE))


def plot_xeb_instance_spread(records: List[XEBRecord]) -> Optional[str]:
    subset = [r for r in records if r.N == 40]
    if not subset:
        return None
    grouped = defaultdict(list)
    for rec in subset:
        grouped[rec.d].append(rec.xeb_fidelity)
    depths = sorted(grouped)
    data = [grouped[d] for d in depths]

    plt.figure(figsize=(8, 4.5))
    plt.boxplot(data, labels=depths, showfliers=True)
    plt.axhline(0.0, linestyle="--", color="black", linewidth=1)
    plt.xlabel("Circuit depth d")
    plt.ylabel("Per-instance linear XEB fidelity")
    plt.title("Distribution of N=40 XEB fidelities across instances")
    plt.grid(axis="y", alpha=0.25)
    out = IMAGES_DIR / "xeb_instance_spread_N40.png"
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    plt.close()
    return str(out.relative_to(WORKSPACE))


def main() -> None:
    xeb_records, missing_xeb = analyze_xeb()
    mb_records, missing_mb = analyze_mb()

    xeb_aggregates = aggregate(xeb_records, protocol="XEB", metric="xeb_fidelity")
    mb_aggregates = aggregate(mb_records, protocol="MB", metric="ideal_success_prob")

    write_csv(
        OUTPUTS_DIR / "xeb_instance_results.csv",
        [asdict(r) for r in xeb_records],
        list(asdict(xeb_records[0]).keys()) if xeb_records else [f.name for f in XEBRecord.__dataclass_fields__.values()],
    )
    write_csv(
        OUTPUTS_DIR / "mb_instance_results.csv",
        [asdict(r) for r in mb_records],
        list(asdict(mb_records[0]).keys()) if mb_records else [f.name for f in MBRecord.__dataclass_fields__.values()],
    )
    write_csv(
        OUTPUTS_DIR / "aggregate_results.csv",
        [asdict(r) for r in (xeb_aggregates + mb_aggregates)],
        list(asdict((xeb_aggregates + mb_aggregates)[0]).keys()) if (xeb_aggregates + mb_aggregates) else [f.name for f in AggregateRecord.__dataclass_fields__.values()],
    )
    write_csv(
        OUTPUTS_DIR / "missing_pairs.csv",
        [asdict(r) for r in (missing_xeb + missing_mb)],
        list(asdict((missing_xeb + missing_mb)[0]).keys()) if (missing_xeb + missing_mb) else [f.name for f in MissingPair.__dataclass_fields__.values()],
    )

    figure_paths = {
        "data_overview": plot_data_overview(xeb_records, mb_records, missing_xeb),
        "xeb_depth_N40": plot_xeb_depth(xeb_records),
        "xeb_size_d12": plot_xeb_size(xeb_records),
        "xeb_instance_spread_N40": plot_xeb_instance_spread(xeb_records),
        "mb_depth": plot_mb_depth(mb_records),
    }

    summary = {
        "workspace": str(WORKSPACE),
        "xeb_records": len(xeb_records),
        "mb_records": len(mb_records),
        "missing_xeb_pairs": len(missing_xeb),
        "missing_mb_pairs": len(missing_mb),
        "xeb_settings": sorted({(r.N, r.d) for r in xeb_records}),
        "mb_settings": sorted({(r.N, r.d) for r in mb_records}),
        "figures": figure_paths,
        "headline_metrics": {},
    }

    xeb_by_setting = {(r.N, r.d): r for r in xeb_aggregates}
    mb_by_setting = {(r.N, r.d): r for r in mb_aggregates}
    if (40, 12) in xeb_by_setting:
        rec = xeb_by_setting[(40, 12)]
        summary["headline_metrics"]["xeb_N40_d12_mean"] = rec.mean
        summary["headline_metrics"]["xeb_N40_d12_sem_across_instances"] = rec.sem
    if xeb_aggregates:
        best = max(xeb_aggregates, key=lambda r: r.mean)
        worst = min(xeb_aggregates, key=lambda r: r.mean)
        summary["headline_metrics"]["best_xeb_setting"] = asdict(best)
        summary["headline_metrics"]["worst_xeb_setting"] = asdict(worst)
    if mb_aggregates:
        best_mb = max(mb_aggregates, key=lambda r: r.mean)
        summary["headline_metrics"]["best_mb_setting"] = asdict(best_mb)
    if (40, 12) in mb_by_setting:
        rec = mb_by_setting[(40, 12)]
        summary["headline_metrics"]["mb_N40_d12_mean_success"] = rec.mean

    with (OUTPUTS_DIR / "analysis_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    notes = [
        "This run computes linear XEB fidelities from matched counts and ideal probabilities,",
        "and MB ideal-bitstring success probabilities from the MB verification files.",
        "XEB uncertainty is the standard error of the counts-weighted per-shot estimator x = 2^N p(x) - 1.",
        "MB uncertainty is the per-instance binomial standard error with an additional 68% Wilson interval.",
        "Some XEB count files lack matching amplitude files; those are listed in outputs/missing_pairs.csv.",
    ]
    (OUTPUTS_DIR / "analysis_notes.txt").write_text("\n".join(notes) + "\n", encoding="utf-8")

    print("Analysis complete.")
    print(f"XEB records: {len(xeb_records)}")
    print(f"MB records: {len(mb_records)}")
    print(f"Missing XEB amplitude pairs: {len(missing_xeb)}")


if __name__ == "__main__":
    main()
