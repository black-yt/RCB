"""21-item data consistency check across ResearchClawBench and ResearchClawBench-Home.

Standalone script — no imports from ResearchClawBench.
Configure RCB and HOME below if repos are not siblings.
"""
import json
import os
import re
import hashlib
import subprocess
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths — adjust if repos are not siblings
# ---------------------------------------------------------------------------
HOME = Path(__file__).resolve().parent
RCB = HOME.parent / "ResearchClawBench"

TASKS = RCB / "tasks"
WS = RCB / "workspaces"
HOME_DATA = HOME / "data"

errors = []
warnings = []


def err(msg): errors.append(msg); print(f"  X {msg}")
def warn(msg): warnings.append(msg); print(f"  ! {msg}")
def ok(msg): print(f"  OK {msg}")


def main():
    if not TASKS.exists():
        print(f"ERROR: ResearchClawBench not found at: {RCB}")
        print("Edit RCB in this script to point to your ResearchClawBench repo.")
        raise SystemExit(1)

    # ============================================================
    # A. TASK LEVEL
    # ============================================================
    print("=" * 60)
    print("A. TASK LEVEL CHECKS")
    print("=" * 60)

    # 1. Task count consistency
    print("\n[1] Task count consistency across locations")
    rcb_tasks = sorted([d.name for d in TASKS.iterdir() if d.is_dir() and "_" in d.name])
    home_tasks = sorted([d.name for d in (HOME_DATA / "tasks").iterdir() if d.is_dir()]) if (HOME_DATA / "tasks").exists() else []

    home_tasks_json = {}
    if (HOME_DATA / "tasks.json").exists():
        with open(HOME_DATA / "tasks.json", encoding="utf-8") as f:
            home_tasks_json = json.load(f)
        home_tasks_from_json = sorted([t for ts in home_tasks_json.values() for t in ts])
    else:
        home_tasks_from_json = []

    print(f"  RCB/tasks: {len(rcb_tasks)}, Home dirs: {len(home_tasks)}, Home tasks.json: {len(home_tasks_from_json)}")
    if set(rcb_tasks) == set(home_tasks) == set(home_tasks_from_json):
        ok(f"All locations have same {len(rcb_tasks)} tasks")
    else:
        diff1 = set(rcb_tasks) - set(home_tasks)
        diff2 = set(home_tasks) - set(rcb_tasks)
        if diff1: err(f"In RCB but not Home dirs: {diff1}")
        if diff2: err(f"In Home dirs but not RCB: {diff2}")
        diff3 = set(rcb_tasks) - set(home_tasks_from_json)
        diff4 = set(home_tasks_from_json) - set(rcb_tasks)
        if diff3: err(f"In RCB but not Home tasks.json: {diff3}")
        if diff4: err(f"In Home tasks.json but not RCB: {diff4}")

    domains = defaultdict(list)
    for t in rcb_tasks:
        domains[t.rsplit("_", 1)[0]].append(t)
    print(f"  Domains: {len(domains)} -- " + ", ".join(f"{d}({len(v)})" for d, v in sorted(domains.items())))

    # Per-task checks (2-10)
    print("\n[2-10] Per-task detailed checks")
    max_path_len = 0
    max_path_str = ""

    for task_id in rcb_tasks:
        task_dir = TASKS / task_id
        home_task_dir = HOME_DATA / "tasks" / task_id

        # Load RCB task_info
        info_path = task_dir / "task_info.json"
        if not info_path.exists():
            err(f"{task_id}: task_info.json missing")
            continue
        with open(info_path, encoding="utf-8") as f:
            info = json.load(f)

        # [2] task_info.json consistency
        home_info_path = home_task_dir / "info.json"
        if home_info_path.exists():
            with open(home_info_path, encoding="utf-8") as f:
                home_info = json.load(f)
            if info != home_info:
                err(f"{task_id}: task_info.json differs from Home info.json")
        else:
            err(f"{task_id}: Home info.json missing")

        # [3] Data file paths valid
        for d in info.get("data", []):
            p = d.get("path", "")
            if not p.startswith("./data/"):
                err(f"{task_id}: path '{p}' doesn't start with ./data/")
            else:
                rel = p[2:]  # strip ./
                actual = task_dir / rel
                if not actual.exists():
                    err(f"{task_id}: data file not found: {rel}")

        # [4] No stale /tasks/ paths in info
        info_str = json.dumps(info)
        stale = re.findall(r'\./tasks/\w+', info_str)
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

        # [6] Data files vs source
        src_data = task_dir / "data"
        if src_data.exists():
            sum(1 for f in src_data.rglob("*") if f.is_file())

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
            size_map = defaultdict(list)
            for f in rw_files:
                size_map[f.stat().st_size].append(f)
            for size, files in size_map.items():
                if len(files) > 1:
                    hashes = defaultdict(list)
                    for f in files:
                        h = hashlib.md5(f.read_bytes()).hexdigest()
                        hashes[h].append(f.name)
                    for h, dups in hashes.items():
                        if len(dups) > 1:
                            err(f"{task_id}: related_work duplicate files: {dups}")

        # [9] target_study exists
        ts = task_dir / "target_study"
        if not ts.exists():
            err(f"{task_id}: target_study/ missing")
        else:
            if not (ts / "checklist.json").exists():
                err(f"{task_id}: target_study/checklist.json missing")
            paper_pdfs = list(ts.glob("paper*.pdf"))
            if not paper_pdfs:
                err(f"{task_id}: target_study/paper*.pdf missing")

        # [10] Check data descriptions for path references
        for d in info.get("data", []):
            desc = d.get("description", "")
            path_refs = re.findall(r'(?:\./|/)tasks/[\w/.\-]+', desc)
            for pr in path_refs:
                err(f"{task_id}: description contains stale path: {pr}")

        task_desc = info.get("task", "")
        path_refs = re.findall(r'(?:\./|/)tasks/[\w/.\-]+', task_desc)
        for pr in path_refs:
            err(f"{task_id}: task description contains stale path: {pr}")

        # Description path validity: check ./data/ refs in descriptions
        for d in info.get("data", []):
            desc = d.get("description", "")
            data_refs = re.findall(r'\./data/[\w/.\-]+', desc)
            for dr in data_refs:
                rel = dr[2:]
                if not (task_dir / rel).exists():
                    err(f"{task_id}: description references non-existent: {dr}")

        # Path length tracking
        for f in task_dir.rglob("*"):
            plen = len(str(f.resolve()))
            if plen > max_path_len:
                max_path_len = plen
                max_path_str = str(f.resolve())

    # [11] Path lengths
    print("\n[11] Path length statistics")
    for f in HOME_DATA.rglob("*"):
        plen = len(str(f.resolve()))
        if plen > max_path_len:
            max_path_len = plen
            max_path_str = str(f.resolve())

    if WS.exists():
        for f in WS.rglob("*"):
            plen = len(str(f.resolve()))
            if plen > max_path_len:
                max_path_len = plen
                max_path_str = str(f.resolve())

    print(f"  Max path length: {max_path_len} chars")
    print(f"  Max path: ...{max_path_str[-80:]}")
    if max_path_len > 260:
        err(f"Path exceeds Windows 260 limit: {max_path_len}")
    elif max_path_len > 240:
        warn(f"Path approaching Windows 260 limit: {max_path_len}")
    else:
        ok(f"All paths within safe range ({max_path_len} max)")

    # ============================================================
    # B. RUN LEVEL
    # ============================================================
    print("\n" + "=" * 60)
    print("B. RUN LEVEL CHECKS")
    print("=" * 60)

    # [12] Workspace metadata
    print("\n[12] Workspace metadata completeness")
    run_dirs = []
    if WS.exists():
        run_dirs = [d for d in WS.iterdir() if d.is_dir()]
        stuck = []
        no_meta = []
        for rd in run_dirs:
            meta_p = rd / "_meta.json"
            if not meta_p.exists():
                no_meta.append(rd.name)
                continue
            with open(meta_p, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("status") == "running":
                stuck.append(rd.name)
        if no_meta: err(f"Runs without _meta.json: {no_meta}")
        if stuck: err(f"Runs stuck as 'running': {stuck}")
        if not no_meta and not stuck:
            ok(f"All {len(run_dirs)} runs have valid _meta.json, none stuck")
    else:
        ok("No workspaces directory")

    # [13] Score file validity
    print("\n[13] Score file validity")
    score_issues = []
    scored_count = 0
    for rd in run_dirs:
        score_p = rd / "_score.json"
        if score_p.exists():
            scored_count += 1
            try:
                with open(score_p, encoding="utf-8") as f:
                    sc = json.load(f)
                if "total_score" not in sc:
                    score_issues.append(f"{rd.name}: missing total_score")
            except json.JSONDecodeError:
                score_issues.append(f"{rd.name}: invalid JSON")
    if score_issues:
        for si in score_issues: err(si)
    else:
        ok(f"All {scored_count} score files valid")

    # [14] Runs data consistency with current tasks
    print("\n[14] Runs data consistency with current tasks")
    run_task_issues = []
    for rd in run_dirs:
        meta_p = rd / "_meta.json"
        if not meta_p.exists():
            continue
        with open(meta_p, encoding="utf-8") as f:
            meta = json.load(f)
        tid = meta.get("task_id", "")
        if tid not in rcb_tasks:
            run_task_issues.append(f"{rd.name}: task_id '{tid}' not in current tasks")
            continue
        # Check if run's data/ matches task's data/
        run_data = rd / "data"
        task_data = TASKS / tid / "data"
        if run_data.exists() and task_data.exists():
            run_files = {}
            for f in run_data.rglob("*"):
                if f.is_file():
                    run_files[str(f.relative_to(run_data))] = f.stat().st_size
            task_files = {}
            for f in task_data.rglob("*"):
                if f.is_file():
                    task_files[str(f.relative_to(task_data))] = f.stat().st_size
            if run_files != task_files:
                missing = set(task_files) - set(run_files)
                extra = set(run_files) - set(task_files)
                size_diff = {k for k in run_files if k in task_files and run_files[k] != task_files[k]}
                details = []
                if missing: details.append(f"missing: {list(missing)[:3]}")
                if extra: details.append(f"extra: {list(extra)[:3]}")
                if size_diff: details.append(f"size differs: {list(size_diff)[:3]}")
                run_task_issues.append(f"{rd.name} (task={tid}): data/ mismatch -- {'; '.join(details)}")

    if run_task_issues:
        for ri in run_task_issues: err(ri)
    else:
        ok(f"All {len(run_dirs)} runs reference valid tasks with matching data")

    # ============================================================
    # C. STATIC SITE (HOME)
    # ============================================================
    print("\n" + "=" * 60)
    print("C. STATIC SITE CHECKS")
    print("=" * 60)

    # [15] runs_index.json
    print("\n[15] runs_index.json consistency")
    if (HOME_DATA / "runs_index.json").exists():
        with open(HOME_DATA / "runs_index.json", encoding="utf-8") as f:
            runs_index = json.load(f)
        missing_runs = []
        for ri in runs_index:
            ws_path = WS / ri["run_id"]
            if not ws_path.exists():
                missing_runs.append(ri["run_id"])
        if missing_runs:
            err(f"Home runs not in workspaces: {missing_runs}")
        else:
            ok(f"runs_index.json: {len(runs_index)} runs, all exist in workspaces/")
    else:
        warn("runs_index.json not found")

    # [16] leaderboard.json
    print("\n[16] leaderboard.json validity")
    lb_path = HOME_DATA / "leaderboard.json"
    if lb_path.exists():
        with open(lb_path, encoding="utf-8") as f:
            lb = json.load(f)
        required = {"tasks", "agents", "scores", "frontier"}
        missing = required - set(lb.keys())
        if missing:
            err(f"leaderboard.json missing fields: {missing}")
        else:
            ok(f"leaderboard.json: {len(lb['tasks'])} tasks, {len(lb['agents'])} agents")
    else:
        warn("leaderboard.json not found")

    # [17] Static assets sync
    print("\n[17] Static assets sync (app.js, style.css)")
    for fname in ["app.js", "style.css"]:
        rcb_f = RCB / "evaluation" / "static" / fname
        home_f = HOME / "static" / fname
        if rcb_f.exists() and home_f.exists():
            if rcb_f.read_bytes() == home_f.read_bytes():
                ok(f"{fname} matches")
            else:
                err(f"{fname} differs between RCB and Home")
        else:
            if not rcb_f.exists(): err(f"RCB {fname} missing")
            if not home_f.exists(): err(f"Home {fname} missing")

    # [18] STATIC_MODE
    print("\n[18] STATIC_MODE in Home index.html")
    home_index = HOME / "index.html"
    if home_index.exists():
        content = home_index.read_text(encoding="utf-8")
        if "STATIC_MODE = true" in content:
            ok("STATIC_MODE = true found")
        else:
            err("STATIC_MODE = true NOT found")
    else:
        err("Home index.html not found")

    # [19] Exported files exist
    print("\n[19] Exported file accessibility")
    export_issues = 0
    export_checked = 0
    for task_id in rcb_tasks:
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

    # Also check run exports
    run_export_issues = 0
    run_export_checked = 0
    if (HOME_DATA / "runs").exists():
        for run_dir in (HOME_DATA / "runs").iterdir():
            if not run_dir.is_dir():
                continue
            files_json = run_dir / "files.json"
            if not files_json.exists():
                continue
            with open(files_json, encoding="utf-8") as f:
                tree = json.load(f)
            for item in tree:
                if item.get("type") == "file" and item.get("exported"):
                    run_export_checked += 1
                    ws_file = run_dir / "workspace" / item["path"]
                    if not ws_file.exists():
                        run_export_issues += 1
        if run_export_issues == 0:
            ok(f"All {run_export_checked} exported run files accessible")
        else:
            err(f"{run_export_issues} exported run files missing")

    # ============================================================
    # D. CONFIG
    # ============================================================
    print("\n" + "=" * 60)
    print("D. CONFIG CHECKS")
    print("=" * 60)

    # [20] .gitignore
    print("\n[20] .gitignore contents")
    gi = RCB / ".gitignore"
    if gi.exists():
        gi_text = gi.read_text(encoding="utf-8")
        for pattern in [".env", "workspaces/", "__pycache__"]:
            if pattern in gi_text:
                ok(f".gitignore contains '{pattern}'")
            else:
                err(f".gitignore missing '{pattern}'")
    else:
        err(".gitignore not found")

    # [21] No secrets in git
    print("\n[21] Sensitive data check")
    result = subprocess.run(["git", "ls-files", ".env"], capture_output=True, text=True, cwd=str(RCB))
    if result.stdout.strip():
        err(".env is tracked by git!")
    else:
        ok(".env not tracked by git")

    # Check no API keys in tracked files
    result2 = subprocess.run(["git", "ls-files", "--", "*.py"], capture_output=True, text=True, cwd=str(RCB))
    tracked_py = [f for f in result2.stdout.strip().split("\n") if f]
    key_found = False
    for pyf in tracked_py:
        fp = RCB / pyf
        if not fp.exists():
            continue
        for line in fp.read_text(encoding="utf-8", errors="replace").split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "sk-" in stripped and "sk-xxx" not in stripped and len(stripped) > 20:
                if "example" not in stripped.lower():
                    err(f"Possible API key in {pyf}")
                    key_found = True
                    break
    if not key_found:
        ok("No API keys in tracked Python files")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Errors:   {len(errors)}")
    print(f"  Warnings: {len(warnings)}")
    if errors:
        print("\nAll errors:")
        for e in errors:
            print(f"  X {e}")
    if warnings:
        print("\nAll warnings:")
        for w in warnings:
            print(f"  ! {w}")
    if not errors and not warnings:
        print("\n  ALL 21 CHECKS PASSED!")


if __name__ == "__main__":
    main()
