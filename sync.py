"""Sync successful runs from ResearchClawBench to ResearchClawBench-Home static site.

Usage: python sync.py
  - Run from the ResearchClawBench-Home directory
  - Automatically finds the ResearchClawBench repo at ../ResearchClawBench
  - Only syncs completed runs (status=completed) that have reports
  - Copies app.js, style.css, and re-exports all data
  - Then: git add -A && git commit && git push
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Paths
RCB_DIR = Path(__file__).resolve().parent
SRC_DIR = RCB_DIR.parent / "ResearchClawBench"
EVAL_DIR = SRC_DIR / "evaluation"

sys.path.insert(0, str(SRC_DIR))


def sync():
    print("=== ResearchClawBench -> RCB Sync ===")

    if not SRC_DIR.exists():
        print(f"ERROR: Source repo not found at {SRC_DIR}")
        return False

    # 1. Copy shared static files
    print("\n[1/4] Syncing static files...")
    for fname in ["app.js", "style.css", "favicon.svg"]:
        src = EVAL_DIR / "static" / fname
        dst = RCB_DIR / "static" / fname
        if src.exists():
            shutil.copy2(src, dst)
            print(f"  Copied {fname}")
    # Logos
    logos_src = EVAL_DIR / "static" / "logos"
    logos_dst = RCB_DIR / "static" / "logos"
    logos_dst.mkdir(parents=True, exist_ok=True)
    if logos_src.exists():
        for f in logos_src.iterdir():
            shutil.copy2(f, logos_dst / f.name)
        print(f"  Copied {len(list(logos_src.iterdir()))} logos")

    # 2. Run the export script
    print("\n[2/4] Exporting data...")
    result = subprocess.run(
        [sys.executable, str(EVAL_DIR / "export_static.py")],
        cwd=str(SRC_DIR),
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        print("ERROR: Export failed!")
        return False

    # 3. Verify — count successful runs
    print("[3/4] Verifying...")
    runs_index = RCB_DIR / "data" / "runs_index.json"
    if runs_index.exists():
        with open(runs_index, "r", encoding="utf-8") as f:
            runs = json.load(f)
        completed = [r for r in runs if r.get("status") == "completed"]
        scored = [r for r in runs if r.get("total_score") is not None]
        print(f"  Total runs: {len(runs)}")
        print(f"  Completed: {len(completed)}")
        print(f"  Scored: {len(scored)}")
    else:
        print("  No runs exported")

    tasks_file = RCB_DIR / "data" / "tasks.json"
    if tasks_file.exists():
        with open(tasks_file, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        total = sum(len(v) for v in tasks.values())
        print(f"  Tasks: {total} across {len(tasks)} domains")

    # 4. Git commit and push
    print("\n[4/4] Committing and pushing...")
    os.chdir(str(RCB_DIR))

    # Check if there are changes
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if not status.stdout.strip():
        print("  No changes to commit")
        return True

    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(
        ["git", "commit", "-m", f"sync: update from ResearchClawBench ({len(completed)} completed runs)"],
        check=True,
    )
    subprocess.run(["git", "push", "origin", "main"], check=True)
    print("\n=== Sync complete! ===")
    return True


if __name__ == "__main__":
    success = sync()
    sys.exit(0 if success else 1)
