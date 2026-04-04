"""21-item data consistency check across ResearchClawBench and ResearchClawBench-Home.

Standalone script — no imports from ResearchClawBench.
Configure RCB and HOME below if repos are not siblings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — adjust if repos are not siblings
# ---------------------------------------------------------------------------
HOME = Path(__file__).resolve().parent
RCB = HOME.parent / "ResearchClawBench"

TASKS = RCB / "tasks"
WS = RCB / "workspaces"
HOME_DATA = HOME / "data"

SECTION_ORDER = ("tasks", "runs", "static", "config")
SECTION_LABELS = {
    "tasks": "A. TASK LEVEL CHECKS",
    "runs": "B. RUN LEVEL CHECKS",
    "static": "C. STATIC SITE CHECKS",
    "config": "D. CONFIG CHECKS",
}

errors: list[str] = []
warnings: list[str] = []
SECRET_RE = re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9]{10,}")


def log(msg: str = "") -> None:
    print(msg, flush=True)


def err(msg: str) -> None:
    errors.append(msg)
    log(f"  X {msg}")


def warn(msg: str) -> None:
    warnings.append(msg)
    log(f"  ! {msg}")


def ok(msg: str) -> None:
    log(f"  OK {msg}")


def info(msg: str) -> None:
    log(f"  ... {msg}")


def elapsed_str(start_time: float) -> str:
    return f"{time.perf_counter() - start_time:.1f}s"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run cross-repo consistency checks between ResearchClawBench and "
            "ResearchClawBench-Home."
        )
    )
    parser.add_argument(
        "--sections",
        nargs="+",
        choices=SECTION_ORDER,
        default=list(SECTION_ORDER),
        help=(
            "Only run the selected sections. Default: tasks runs static config. "
            "Example: --sections runs static"
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print loop progress every N items. Set to 0 to disable loop progress output.",
    )
    return parser.parse_args(argv)


def ensure_repo_exists() -> None:
    if not TASKS.exists():
        log(f"ERROR: ResearchClawBench not found at: {RCB}")
        log("Edit RCB in this script to point to your ResearchClawBench repo.")
        raise SystemExit(1)


def should_report_progress(index: int, total: int, every: int) -> bool:
    if every <= 0 or total <= 0:
        return False
    return index == 1 or index == total or index % every == 0


def report_progress(label: str, index: int, total: int, loop_start: float, every: int) -> None:
    if should_report_progress(index, total, every):
        info(f"{label}: {index}/{total} ({elapsed_str(loop_start)})")


def begin_section(name: str) -> float:
    log("\n" + "=" * 60)
    log(SECTION_LABELS[name])
    log("=" * 60)
    return time.perf_counter()


def end_section(name: str, start_time: float) -> None:
    info(f"Finished {SECTION_LABELS[name]} in {elapsed_str(start_time)}")


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_rel_size_map(root: Path) -> dict[str, int]:
    files: dict[str, int] = {}
    for path in root.rglob("*"):
        if path.is_file():
            files[str(path.relative_to(root))] = path.stat().st_size
    return files


def scan_max_path(root: Path, current_max: int, current_path: str) -> tuple[int, str]:
    if not root.exists():
        return current_max, current_path
    info(f"Scanning path lengths under {root} ...")
    stack = [os.fspath(root)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    path_str = entry.path
                    plen = len(path_str)
                    if plen > current_max:
                        current_max = plen
                        current_path = path_str
                    if entry.is_dir(follow_symlinks=False):
                        stack.append(path_str)
        except OSError:
            continue
    return current_max, current_path


def run_task_checks(rcb_tasks: list[str], progress_every: int) -> None:
    section_start = begin_section("tasks")

    # 1. Task count consistency
    log("\n[1] Task count consistency across locations")
    home_tasks = (
        sorted(d.name for d in (HOME_DATA / "tasks").iterdir() if d.is_dir())
        if (HOME_DATA / "tasks").exists()
        else []
    )

    if (HOME_DATA / "tasks.json").exists():
        with open(HOME_DATA / "tasks.json", encoding="utf-8") as f:
            home_tasks_json = json.load(f)
        home_tasks_from_json = sorted(t for ts in home_tasks_json.values() for t in ts)
    else:
        home_tasks_json = {}
        home_tasks_from_json = []

    log(
        f"  RCB/tasks: {len(rcb_tasks)}, "
        f"Home dirs: {len(home_tasks)}, "
        f"Home tasks.json: {len(home_tasks_from_json)}"
    )
    if set(rcb_tasks) == set(home_tasks) == set(home_tasks_from_json):
        ok(f"All locations have same {len(rcb_tasks)} tasks")
    else:
        diff1 = set(rcb_tasks) - set(home_tasks)
        diff2 = set(home_tasks) - set(rcb_tasks)
        diff3 = set(rcb_tasks) - set(home_tasks_from_json)
        diff4 = set(home_tasks_from_json) - set(rcb_tasks)
        if diff1:
            err(f"In RCB but not Home dirs: {diff1}")
        if diff2:
            err(f"In Home dirs but not RCB: {diff2}")
        if diff3:
            err(f"In RCB but not Home tasks.json: {diff3}")
        if diff4:
            err(f"In Home tasks.json but not RCB: {diff4}")

    domains: dict[str, list[str]] = defaultdict(list)
    for task_id in rcb_tasks:
        domains[task_id.rsplit("_", 1)[0]].append(task_id)
    log(
        f"  Domains: {len(domains)} -- "
        + ", ".join(f"{d}({len(v)})" for d, v in sorted(domains.items()))
    )

    # 2-10. Per-task checks
    log("\n[2-10] Per-task detailed checks")
    max_path_len = 0
    max_path_str = ""
    task_loop_start = time.perf_counter()

    for index, task_id in enumerate(rcb_tasks, start=1):
        report_progress("Task checks", index, len(rcb_tasks), task_loop_start, progress_every)

        task_dir = TASKS / task_id
        home_task_dir = HOME_DATA / "tasks" / task_id
        info_path = task_dir / "task_info.json"

        if not info_path.exists():
            err(f"{task_id}: task_info.json missing")
            continue

        with open(info_path, encoding="utf-8") as f:
            info_json = json.load(f)

        # [2] task_info.json consistency
        home_info_path = home_task_dir / "info.json"
        if home_info_path.exists():
            with open(home_info_path, encoding="utf-8") as f:
                home_info = json.load(f)
            if info_json != home_info:
                err(f"{task_id}: task_info.json differs from Home info.json")
        else:
            err(f"{task_id}: Home info.json missing")

        # [3] Data file paths valid
        for item in info_json.get("data", []):
            path_str = item.get("path", "")
            if not path_str.startswith("./data/"):
                err(f"{task_id}: path '{path_str}' doesn't start with ./data/")
                continue
            rel = path_str[2:]
            actual = task_dir / rel
            if not actual.exists():
                err(f"{task_id}: data file not found: {rel}")

        # [4] No stale /tasks/ paths in info
        info_str = json.dumps(info_json)
        stale = re.findall(r"\./tasks/\w+", info_str)
        if stale:
            err(f"{task_id}: stale /tasks/ path refs: {stale}")

        # [5] checklist.json consistency
        rcb_cl = task_dir / "target_study" / "checklist.json"
        home_cl = home_task_dir / "checklist.json"
        if rcb_cl.exists() and home_cl.exists():
            with open(rcb_cl, encoding="utf-8") as f:
                cl1 = json.load(f)
            with open(home_cl, encoding="utf-8") as f:
                cl2 = json.load(f)
            if cl1 != cl2:
                err(f"{task_id}: checklist.json differs")
        elif rcb_cl.exists() and not home_cl.exists():
            err(f"{task_id}: Home checklist.json missing")

        # [7] related_work exists and non-empty, PDF naming
        rw = task_dir / "related_work"
        if not rw.exists() or not list(rw.iterdir()):
            err(f"{task_id}: related_work/ missing or empty")
        else:
            pdfs = list(rw.glob("*.pdf"))
            non_paper = [p.name for p in pdfs if not p.name.startswith("paper_")]
            if non_paper:
                warn(f"{task_id}: related_work PDFs not named paper_*: {non_paper}")

        # [8] related_work no duplicate files (by content hash)
        if rw.exists():
            rw_files = [f for f in rw.iterdir() if f.is_file()]
            size_map: dict[int, list[Path]] = defaultdict(list)
            for file_path in rw_files:
                size_map[file_path.stat().st_size].append(file_path)
            for files in size_map.values():
                if len(files) <= 1:
                    continue
                hashes: dict[str, list[str]] = defaultdict(list)
                for file_path in files:
                    hashes[md5_file(file_path)].append(file_path.name)
                for dup_names in hashes.values():
                    if len(dup_names) > 1:
                        err(f"{task_id}: related_work duplicate files: {dup_names}")

        # [9] target_study exists
        ts = task_dir / "target_study"
        if not ts.exists():
            err(f"{task_id}: target_study/ missing")
        else:
            if not (ts / "checklist.json").exists():
                err(f"{task_id}: target_study/checklist.json missing")
            if not list(ts.glob("paper*.pdf")):
                err(f"{task_id}: target_study/paper*.pdf missing")

        # [10] Description path refs
        for item in info_json.get("data", []):
            desc = item.get("description", "")
            path_refs = re.findall(r"(?:\./|/)tasks/[\w/.\-]+", desc)
            for ref in path_refs:
                err(f"{task_id}: description contains stale path: {ref}")

            data_refs = re.findall(r"\./data/[\w/.\-]+", desc)
            for ref in data_refs:
                rel = ref[2:]
                if not (task_dir / rel).exists():
                    err(f"{task_id}: description references non-existent: {ref}")

        task_desc = info_json.get("task", "")
        path_refs = re.findall(r"(?:\./|/)tasks/[\w/.\-]+", task_desc)
        for ref in path_refs:
            err(f"{task_id}: task description contains stale path: {ref}")

    # [11] Path lengths
    log("\n[11] Path length statistics")
    max_path_len, max_path_str = scan_max_path(TASKS, max_path_len, max_path_str)
    max_path_len, max_path_str = scan_max_path(HOME_DATA, max_path_len, max_path_str)
    max_path_len, max_path_str = scan_max_path(WS, max_path_len, max_path_str)

    log(f"  Max path length: {max_path_len} chars")
    log(f"  Max path: ...{max_path_str[-80:]}")
    if max_path_len > 260:
        err(f"Path exceeds Windows 260 limit: {max_path_len}")
    elif max_path_len > 240:
        warn(f"Path approaching Windows 260 limit: {max_path_len}")
    else:
        ok(f"All paths within safe range ({max_path_len} max)")

    end_section("tasks", section_start)


def run_run_checks(rcb_tasks: list[str], progress_every: int) -> None:
    section_start = begin_section("runs")
    rcb_task_set = set(rcb_tasks)

    log("\n[12] Workspace metadata completeness")
    if WS.exists():
        run_dirs = [d for d in WS.iterdir() if d.is_dir()]
        stuck = []
        no_meta = []
        meta_cache: dict[str, dict] = {}
        loop_start = time.perf_counter()
        for index, run_dir in enumerate(run_dirs, start=1):
            report_progress("Run metadata", index, len(run_dirs), loop_start, progress_every)
            meta_path = run_dir / "_meta.json"
            if not meta_path.exists():
                no_meta.append(run_dir.name)
                continue
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            meta_cache[run_dir.name] = meta
            if meta.get("status") == "running":
                stuck.append(run_dir.name)
        if no_meta:
            err(f"Runs without _meta.json: {no_meta}")
        if stuck:
            err(f"Runs stuck as 'running': {stuck}")
        if not no_meta and not stuck:
            ok(f"All {len(run_dirs)} runs have valid _meta.json, none stuck")
    else:
        run_dirs = []
        meta_cache = {}
        ok("No workspaces directory")

    log("\n[13] Score file validity")
    score_issues = []
    scored_count = 0
    score_loop_start = time.perf_counter()
    for index, run_dir in enumerate(run_dirs, start=1):
        report_progress("Score files", index, len(run_dirs), score_loop_start, progress_every)
        score_path = run_dir / "_score.json"
        if not score_path.exists():
            continue
        scored_count += 1
        try:
            with open(score_path, encoding="utf-8") as f:
                score = json.load(f)
            if "total_score" not in score:
                score_issues.append(f"{run_dir.name}: missing total_score")
        except json.JSONDecodeError:
            score_issues.append(f"{run_dir.name}: invalid JSON")
    if score_issues:
        for issue in score_issues:
            err(issue)
    else:
        ok(f"All {scored_count} score files valid")

    log("\n[14] Runs data consistency with current tasks")
    run_task_issues = []
    task_files_cache: dict[str, dict[str, int]] = {}
    run_check_start = time.perf_counter()
    for index, run_dir in enumerate(run_dirs, start=1):
        report_progress("Run/task data compare", index, len(run_dirs), run_check_start, progress_every)
        meta = meta_cache.get(run_dir.name)
        if meta is None:
            meta_path = run_dir / "_meta.json"
            if not meta_path.exists():
                continue
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)

        task_id = meta.get("task_id", "")
        if task_id not in rcb_task_set:
            run_task_issues.append(f"{run_dir.name}: task_id '{task_id}' not in current tasks")
            continue

        run_data = run_dir / "data"
        task_data = TASKS / task_id / "data"
        if not (run_data.exists() and task_data.exists()):
            continue

        run_files = build_rel_size_map(run_data)
        if task_id not in task_files_cache:
            task_files_cache[task_id] = build_rel_size_map(task_data)
        task_files = task_files_cache[task_id]

        if run_files != task_files:
            missing = set(task_files) - set(run_files)
            extra = set(run_files) - set(task_files)
            size_diff = {
                key for key in run_files if key in task_files and run_files[key] != task_files[key]
            }
            details = []
            if missing:
                details.append(f"missing: {list(missing)[:3]}")
            if extra:
                details.append(f"extra: {list(extra)[:3]}")
            if size_diff:
                details.append(f"size differs: {list(size_diff)[:3]}")
            run_task_issues.append(
                f"{run_dir.name} (task={task_id}): data/ mismatch -- {'; '.join(details)}"
            )

    if run_task_issues:
        for issue in run_task_issues:
            err(issue)
    else:
        ok(f"All {len(run_dirs)} runs reference valid tasks with matching data")

    end_section("runs", section_start)


def run_static_checks(rcb_tasks: list[str], progress_every: int) -> None:
    section_start = begin_section("static")

    log("\n[15] runs_index.json consistency")
    runs_index_path = HOME_DATA / "runs_index.json"
    if runs_index_path.exists():
        with open(runs_index_path, encoding="utf-8") as f:
            runs_index = json.load(f)
        missing_runs = []
        loop_start = time.perf_counter()
        for index, item in enumerate(runs_index, start=1):
            report_progress("runs_index", index, len(runs_index), loop_start, progress_every)
            ws_path = WS / item["run_id"]
            if not ws_path.exists():
                missing_runs.append(item["run_id"])
        if missing_runs:
            err(f"Home runs not in workspaces: {missing_runs}")
        else:
            ok(f"runs_index.json: {len(runs_index)} runs, all exist in workspaces/")
    else:
        warn("runs_index.json not found")

    log("\n[16] leaderboard.json validity")
    lb_path = HOME_DATA / "leaderboard.json"
    if lb_path.exists():
        with open(lb_path, encoding="utf-8") as f:
            leaderboard = json.load(f)
        required = {"tasks", "agents", "scores", "frontier"}
        missing = required - set(leaderboard.keys())
        if missing:
            err(f"leaderboard.json missing fields: {missing}")
        else:
            ok(
                f"leaderboard.json: {len(leaderboard['tasks'])} tasks, "
                f"{len(leaderboard['agents'])} agents"
            )
            cell_required = {"score", "run_id", "duration_seconds", "cost_usd", "model", "model_display"}
            cell_errors = 0
            for agent_name, task_scores in leaderboard.get("scores", {}).items():
                if not isinstance(task_scores, dict):
                    err(f"leaderboard.json scores[{agent_name!r}] is not an object")
                    cell_errors += 1
                    continue
                for task_id, entry in task_scores.items():
                    if not isinstance(entry, dict):
                        err(f"leaderboard.json scores[{agent_name!r}][{task_id!r}] is not an object")
                        cell_errors += 1
                        continue
                    missing_cell = cell_required - set(entry.keys())
                    if missing_cell:
                        err(
                            f"leaderboard.json scores[{agent_name!r}][{task_id!r}] "
                            f"missing fields: {sorted(missing_cell)}"
                        )
                        cell_errors += 1
            frontier = leaderboard.get("frontier")
            if not isinstance(frontier, dict):
                err("leaderboard.json frontier is not an object")
                cell_errors += 1
            else:
                missing_frontier = [task_id for task_id in leaderboard.get("tasks", []) if task_id not in frontier]
                if missing_frontier:
                    err(f"leaderboard.json frontier missing tasks: {missing_frontier[:3]}")
                    cell_errors += len(missing_frontier)
            if cell_errors == 0:
                ok("leaderboard.json cell payloads look complete")
    else:
        warn("leaderboard.json not found")

    log("\n[17] Static assets sync (app.js, style.css)")
    for fname in ["app.js", "style.css"]:
        rcb_file = RCB / "evaluation" / "static" / fname
        home_file = HOME / "static" / fname
        if rcb_file.exists() and home_file.exists():
            if rcb_file.read_bytes() == home_file.read_bytes():
                ok(f"{fname} matches")
            else:
                err(f"{fname} differs between RCB and Home")
        else:
            if not rcb_file.exists():
                err(f"RCB {fname} missing")
            if not home_file.exists():
                err(f"Home {fname} missing")

    log("\n[18] STATIC_MODE in Home index.html")
    home_index = HOME / "index.html"
    if home_index.exists():
        content = home_index.read_text(encoding="utf-8")
        if "STATIC_MODE = true" in content:
            ok("STATIC_MODE = true found")
        else:
            err("STATIC_MODE = true NOT found")
    else:
        err("Home index.html not found")

    log("\n[19] Exported file accessibility")
    export_issues = 0
    export_checked = 0
    task_export_start = time.perf_counter()
    for index, task_id in enumerate(rcb_tasks, start=1):
        report_progress("Task exports", index, len(rcb_tasks), task_export_start, progress_every)
        files_json = HOME_DATA / "tasks" / task_id / "files.json"
        if not files_json.exists():
            continue
        with open(files_json, encoding="utf-8") as f:
            tree = json.load(f)
        for item in tree:
            if item.get("type") == "file" and item.get("exported"):
                export_checked += 1
                ws_file = HOME_DATA / "tasks" / task_id / "workspace" / item["path"]
                if not ws_file.exists():
                    export_issues += 1
                    if export_issues <= 3:
                        err(f"{task_id}: exported file missing: {item['path']}")

    if export_issues == 0:
        ok(f"All {export_checked} exported task files accessible")
    elif export_issues > 3:
        err(f"... and {export_issues - 3} more missing exported files")

    run_export_issues = 0
    run_export_checked = 0
    runs_dir = HOME_DATA / "runs"
    if runs_dir.exists():
        run_export_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
        run_export_start = time.perf_counter()
        for index, run_dir in enumerate(run_export_dirs, start=1):
            report_progress(
                "Run exports", index, len(run_export_dirs), run_export_start, progress_every
            )
            files_json = run_dir / "files.json"
            if not files_json.exists():
                continue

            run_task_id = None
            data_json = run_dir / "data.json"
            if data_json.exists():
                try:
                    with open(data_json, encoding="utf-8") as f:
                        run_task_id = json.load(f).get("task_id")
                except (json.JSONDecodeError, OSError):
                    pass

            with open(files_json, encoding="utf-8") as f:
                tree = json.load(f)

            for item in tree:
                if item.get("type") != "file" or not item.get("exported"):
                    continue
                run_export_checked += 1
                if item.get("shared"):
                    if not run_task_id:
                        run_export_issues += 1
                        continue
                    ws_file = HOME_DATA / "tasks" / run_task_id / "workspace" / item["path"]
                else:
                    ws_file = run_dir / "workspace" / item["path"]
                if not ws_file.exists():
                    run_export_issues += 1

        if run_export_issues == 0:
            ok(f"All {run_export_checked} exported run files accessible")
        else:
            err(f"{run_export_issues} exported run files missing")

    end_section("static", section_start)


def run_config_checks() -> None:
    section_start = begin_section("config")

    log("\n[20] .gitignore contents")
    gitignore = RCB / ".gitignore"
    if gitignore.exists():
        gitignore_text = gitignore.read_text(encoding="utf-8")
        for pattern in [".env", "workspaces/", "__pycache__"]:
            if pattern in gitignore_text:
                ok(f".gitignore contains '{pattern}'")
            else:
                err(f".gitignore missing '{pattern}'")
    else:
        err(".gitignore not found")

    log("\n[21] Sensitive data check")
    result = subprocess.run(
        ["git", "ls-files", ".env"],
        capture_output=True,
        text=True,
        cwd=str(RCB),
        check=False,
    )
    if result.stdout.strip():
        err(".env is tracked by git!")
    else:
        ok(".env not tracked by git")

    result2 = subprocess.run(
        ["git", "ls-files", "--", "*.py"],
        capture_output=True,
        text=True,
        cwd=str(RCB),
        check=False,
    )
    tracked_py = [f for f in result2.stdout.strip().split("\n") if f]
    key_found = False
    loop_start = time.perf_counter()
    for index, pyf in enumerate(tracked_py, start=1):
        report_progress("Tracked Python secret scan", index, len(tracked_py), loop_start, 100)
        file_path = RCB / pyf
        if not file_path.exists():
            continue
        for line in file_path.read_text(encoding="utf-8", errors="replace").split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            lowered = stripped.lower()
            if "sk-xxx" in lowered or "example" in lowered:
                continue
            if SECRET_RE.search(stripped):
                err(f"Possible API key in {pyf}")
                key_found = True
                break
    if not key_found:
        ok("No API keys in tracked Python files")

    end_section("config", section_start)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ensure_repo_exists()

    selected_sections = [name for name in SECTION_ORDER if name in set(args.sections)]
    total_start = time.perf_counter()
    sys.stdout.reconfigure(line_buffering=True)

    info(
        "Running sections: "
        + ", ".join(selected_sections)
        + f" | progress every {args.progress_every} item(s)"
    )

    rcb_tasks = sorted(d.name for d in TASKS.iterdir() if d.is_dir() and "_" in d.name)

    if "tasks" in selected_sections:
        run_task_checks(rcb_tasks, args.progress_every)
    if "runs" in selected_sections:
        run_run_checks(rcb_tasks, args.progress_every)
    if "static" in selected_sections:
        run_static_checks(rcb_tasks, args.progress_every)
    if "config" in selected_sections:
        run_config_checks()

    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    log(f"  Sections: {', '.join(selected_sections)}")
    log(f"  Errors:   {len(errors)}")
    log(f"  Warnings: {len(warnings)}")
    log(f"  Elapsed:  {elapsed_str(total_start)}")
    if errors:
        log("\nAll errors:")
        for message in errors:
            log(f"  X {message}")
    if warnings:
        log("\nAll warnings:")
        for message in warnings:
            log(f"  ! {message}")
    if not errors and not warnings:
        if selected_sections == list(SECTION_ORDER):
            log("\n  ALL 21 CHECKS PASSED!")
        else:
            log("\n  ALL SELECTED CHECKS PASSED!")


if __name__ == "__main__":
    main()
