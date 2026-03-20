"""Export evaluation data from ResearchClawBench to static JSON for GitHub Pages.

Standalone script — no imports from ResearchClawBench.
Configure RCB_SOURCE below to point to your ResearchClawBench repo root.
"""

import json
import re
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — adjust RCB_SOURCE if repos are not siblings
# ---------------------------------------------------------------------------
HOME_DIR = Path(__file__).resolve().parent
DATA_DIR = HOME_DIR / "data"

RCB_SOURCE = HOME_DIR.parent / "ResearchClawBench"
TASKS_DIR = RCB_SOURCE / "tasks"
WORKSPACES_DIR = RCB_SOURCE / "workspaces"
STATIC_SRC = RCB_SOURCE / "evaluation" / "static"

# Viewable text extensions (must match app.js)
TEXT_EXTS = {
    '.txt', '.md', '.py', '.js', '.json', '.jsonl', '.csv', '.tsv',
    '.yml', '.yaml', '.sh', '.bash', '.r', '.html', '.css', '.xml',
    '.ini', '.cfg', '.conf', '.toml', '.log', '.dat', '.tex', '.bib',
    '.sql', '.c', '.cpp', '.h', '.java', '.go', '.rs', '.jl', '.m', '.ipynb',
}
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg'}


# ---------------------------------------------------------------------------
# Helpers (self-contained, no RCB imports)
# ---------------------------------------------------------------------------

def _list_tasks():
    """Return sorted task IDs from TASKS_DIR."""
    if not TASKS_DIR.exists():
        return []
    return sorted(
        d.name for d in TASKS_DIR.iterdir()
        if d.is_dir() and (d / "task_info.json").exists()
    )


def _list_tasks_grouped():
    """Return {domain: [task_id, ...]}."""
    groups = {}
    for task_id in _list_tasks():
        m = re.match(r"([A-Za-z]+)_", task_id)
        domain = m.group(1) if m else "Other"
        groups.setdefault(domain, []).append(task_id)
    return groups


def _load_task_info(task_id):
    with open(TASKS_DIR / task_id / "task_info.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _load_checklist(task_id):
    with open(TASKS_DIR / task_id / "target_study" / "checklist.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _list_runs():
    """List all runs from WORKSPACES_DIR."""
    if not WORKSPACES_DIR.exists():
        return []
    runs = []
    for d in sorted(WORKSPACES_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        meta_path = d / "_meta.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        runs.append({
            "run_id": d.name,
            "task_id": meta.get("task_id"),
            "timestamp": meta.get("timestamp"),
            "status": meta.get("status", "unknown"),
            "agent_name": meta.get("agent_name", ""),
            "model": meta.get("model", ""),
            "duration_seconds": meta.get("duration_seconds"),
            "workspace": str(d),
        })
    return runs


def _get_run_workspace(run_id):
    ws = WORKSPACES_DIR / run_id
    return ws if ws.is_dir() else None


def _build_file_tree(root, prefix=""):
    """Build flat file tree list for a directory."""
    skip_names = {"_meta.json", "_agent_output.jsonl", "_score.json", ".claude"}
    tree = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return tree
    for entry in entries:
        if entry.name.startswith(".") or entry.name in skip_names:
            continue
        rel = f"{prefix}/{entry.name}" if prefix else entry.name
        if entry.is_dir():
            tree.append({"name": entry.name, "path": rel, "type": "directory"})
            tree.extend(_build_file_tree(entry, rel))
        else:
            try:
                stat = entry.stat()
            except OSError:
                continue
            tree.append({
                "name": entry.name, "path": rel, "type": "file",
                "size": stat.st_size, "mtime": stat.st_mtime,
            })
    return tree


def _build_instructions(task_info):
    """Build INSTRUCTIONS.md content from task_info dict."""
    task_desc = task_info.get("task", "")
    data_parts = []
    for d in task_info.get("data", []):
        ws_path = d.get("path", "").lstrip("./")
        data_type = d.get("type", "")
        type_str = f" [{data_type}]" if data_type else ""
        data_parts.append(f"- **{d['name']}**{type_str} (`{ws_path}`): {d.get('description', '')}")
    data_text = "\n".join(data_parts) if data_parts else "No specific data files."

    return f"""# Research Task

## Task Description
{task_desc}

## Available Data Files
{data_text}

## Workspace Layout
- `data/` — Input datasets (read-only, do not modify)
- `related_work/` — Reference papers and materials
- `code/` — Write your analysis code here
- `outputs/` — Save intermediate results
- `report/` — Write your final research report here
- `report/images/` — Save all report figures here

## Deliverables
1. Write analysis code in `code/` that processes the data
2. Save intermediate outputs to `outputs/`
3. Write a comprehensive research report as `report/report.md`
   - Include methodology, results, and discussion
   - Use proper academic writing style
   - **You MUST include figures in your report.** Generate plots, charts, and visualizations that support your analysis
   - Save all report figures to `report/images/` and reference them in the report using relative paths: `images/figure_name.png`
   - Include at least: data overview plots, main result figures, and comparison/validation plots

## Guidelines
- Install any needed Python packages via pip
- Use matplotlib/seaborn for visualization
- Ensure all code is reproducible
- Document your approach clearly in the report
"""


# ---------------------------------------------------------------------------
# Export functions
# ---------------------------------------------------------------------------

def export_tasks():
    grouped = _list_tasks_grouped()
    (DATA_DIR / "tasks").mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "tasks.json", "w", encoding="utf-8") as f:
        json.dump(grouped, f, indent=2)

    for domain, task_ids in grouped.items():
        for task_id in task_ids:
            task_dir = DATA_DIR / "tasks" / task_id
            task_dir.mkdir(parents=True, exist_ok=True)
            info = _load_task_info(task_id)
            with open(task_dir / "info.json", "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)
            try:
                checklist = _load_checklist(task_id)
                with open(task_dir / "checklist.json", "w", encoding="utf-8") as f:
                    json.dump(checklist, f, indent=2)
            except FileNotFoundError:
                pass
            # Copy checklist images
            images_dir = TASKS_DIR / task_id / "target_study" / "images"
            if images_dir.exists():
                dst = task_dir / "images"
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(images_dir, dst)

            # Copy target paper PDF (skip if > 10MB)
            target_study = TASKS_DIR / task_id / "target_study"
            for pdf in target_study.glob("paper*.pdf"):
                if pdf.stat().st_size < 10 * 1024 * 1024:
                    shutil.copy2(pdf, task_dir / "paper.pdf")
                    break

            # Generate task file tree
            src_task = TASKS_DIR / task_id
            instructions_text = _build_instructions(info)

            tree = []
            top_dirs = {}
            for subdir in ["data", "related_work"]:
                sub_path = src_task / subdir
                if sub_path.exists():
                    top_dirs[subdir] = _build_file_tree(sub_path, subdir)
            for d in ["code", "outputs", "report"]:
                if d not in top_dirs:
                    top_dirs[d] = []
            top_dirs.setdefault("report", []).insert(0, {"name": "images", "path": "report/images", "type": "directory"})
            for name in sorted(top_dirs.keys()):
                tree.append({"name": name, "path": name, "type": "directory"})
                tree.extend(top_dirs[name])
            tree.append({"name": "INSTRUCTIONS.md", "path": "INSTRUCTIONS.md", "type": "file", "size": len(instructions_text)})

            with open(task_dir / "files.json", "w", encoding="utf-8") as f:
                json.dump(tree, f, indent=2)

            # Write INSTRUCTIONS.md
            with open(task_dir / "INSTRUCTIONS.md", "w", encoding="utf-8") as f:
                f.write(instructions_text)

            # Copy viewable task files preserving structure
            workspace_dst = task_dir / "workspace"
            if workspace_dst.exists():
                shutil.rmtree(workspace_dst, ignore_errors=True)
            exported_paths = set()
            for item in tree:
                if item["type"] != "file":
                    continue
                if item["path"] == "INSTRUCTIONS.md":
                    dst_file = workspace_dst / "INSTRUCTIONS.md"
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(dst_file, "w", encoding="utf-8") as f:
                        f.write(instructions_text)
                    exported_paths.add(item["path"])
                    continue
                src_file = src_task / item["path"]
                if not src_file.exists():
                    continue
                ext = src_file.suffix.lower()
                max_size = 15 * 1024 * 1024 if ext == '.pdf' else 2 * 1024 * 1024
                if (ext in TEXT_EXTS or ext in IMG_EXTS or ext == '.pdf') and src_file.stat().st_size < max_size:
                    dst_file = workspace_dst / item["path"]
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                    exported_paths.add(item["path"])

            # Mark exported files in the tree
            for item in tree:
                if item["type"] == "file":
                    item["exported"] = item["path"] in exported_paths

            with open(task_dir / "files.json", "w", encoding="utf-8") as f:
                json.dump(tree, f, indent=2)

    print(f"Exported {sum(len(v) for v in grouped.values())} tasks")


def export_runs():
    runs = _list_runs()
    runs_dir = DATA_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    exported = []
    for run in runs:
        ws = _get_run_workspace(run["run_id"])
        if not ws:
            continue
        meta_path = ws / "_meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Only export completed runs
        if meta.get("status") != "completed":
            continue

        run_out_dir = runs_dir / run["run_id"]
        run_out_dir.mkdir(parents=True, exist_ok=True)

        run_data = {
            "run_id": run["run_id"],
            "task_id": meta.get("task_id"),
            "timestamp": meta.get("timestamp"),
            "status": meta.get("status"),
            "agent_name": meta.get("agent_name", ""),
            "model": meta.get("model", ""),
            "duration_seconds": meta.get("duration_seconds"),
        }

        # Score
        score_path = ws / "_score.json"
        if score_path.exists():
            with open(score_path, "r", encoding="utf-8") as f:
                run_data["score"] = json.load(f)

        # Report text
        report_path = ws / "report" / "report.md"
        if report_path.exists():
            run_data["report"] = report_path.read_text(encoding="utf-8", errors="replace")

        # Agent output (last 500 lines; prefer JSON lines if available)
        MAX_OUTPUT_LINES = 500
        for output_name in ["_agent_output.jsonl", "_claude_output.jsonl"]:
            output_path = ws / output_name
            if output_path.exists():
                all_lines = []
                json_lines = []
                with open(output_path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        all_lines.append(line)
                        if line.startswith('{'):
                            try:
                                json.loads(line)
                                json_lines.append(line)
                            except json.JSONDecodeError:
                                pass
                source = json_lines if len(json_lines) > 10 else all_lines
                exported_lines = source[-MAX_OUTPUT_LINES:] if len(source) > MAX_OUTPUT_LINES else source
                with open(run_out_dir / "output.json", "w", encoding="utf-8") as f:
                    json.dump(exported_lines, f)
                break

        # File tree
        tree = _build_file_tree(ws)
        with open(run_out_dir / "files.json", "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2)

        # Copy viewable files preserving directory structure
        files_dst = run_out_dir / "workspace"
        if files_dst.exists():
            shutil.rmtree(files_dst)
        run_exported = set()
        for item in tree:
            if item["type"] != "file":
                continue
            src = ws / item["path"]
            if not src.exists():
                continue
            ext = src.suffix.lower()
            if ext in TEXT_EXTS or ext in IMG_EXTS or ext == '.pdf':
                max_size = 15 * 1024 * 1024 if ext == '.pdf' else 2 * 1024 * 1024
                if src.stat().st_size > max_size:
                    continue
                dst = files_dst / item["path"]
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                run_exported.add(item["path"])

        # Mark exported files in tree
        for item in tree:
            if item["type"] == "file":
                item["exported"] = item["path"] in run_exported
        with open(run_out_dir / "files.json", "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2)

        # Save run data
        with open(run_out_dir / "data.json", "w", encoding="utf-8") as f:
            json.dump(run_data, f, indent=2, ensure_ascii=False)

        exported.append(run_data)

    # runs_index.json
    index = [{
        "run_id": r["run_id"],
        "task_id": r["task_id"],
        "timestamp": r["timestamp"],
        "status": r["status"],
        "agent_name": r["agent_name"],
        "model": r["model"],
        "duration_seconds": r.get("duration_seconds"),
        "total_score": r.get("score", {}).get("total_score"),
    } for r in exported]
    with open(DATA_DIR / "runs_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Exported {len(exported)} runs")


def export_leaderboard():
    runs = _list_runs()
    best = {}
    for run in runs:
        ws = _get_run_workspace(run["run_id"])
        if not ws:
            continue
        score_path = ws / "_score.json"
        if not score_path.exists():
            continue
        try:
            with open(score_path, "r", encoding="utf-8") as f:
                score_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        task_id = run["task_id"]
        agent = score_data.get("agent_name", run.get("agent_name", "Unknown"))
        total = score_data.get("total_score", 0)
        key = (task_id, agent)
        if key not in best or total > best[key]["score"]:
            best[key] = {"score": total, "run_id": run["run_id"]}

    tasks_set, agents_set = set(), set()
    for (t, a) in best:
        tasks_set.add(t)
        agents_set.add(a)
    tasks_list, agents_list = sorted(tasks_set), sorted(agents_set)
    scores = {a: {t: best[(t, a)] for t in tasks_list if (t, a) in best} for a in agents_list}
    frontier = {t: max((best[(t, a)]["score"] for a in agents_list if (t, a) in best), default=0) for t in tasks_list}

    with open(DATA_DIR / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump({"tasks": tasks_list, "agents": agents_list, "scores": scores, "frontier": frontier}, f, indent=2)
    print(f"Exported leaderboard: {len(tasks_list)} tasks, {len(agents_list)} agents")


def copy_static():
    dst = HOME_DIR / "static"
    for d in ["logos"]:
        (dst / d).mkdir(parents=True, exist_ok=True)
        for f in (STATIC_SRC / d).iterdir():
            shutil.copy2(f, dst / d / f.name)
    shutil.copy2(STATIC_SRC / "favicon.svg", dst / "favicon.svg")
    print("Copied static assets")


if __name__ == "__main__":
    if not TASKS_DIR.exists():
        print(f"ERROR: RCB_SOURCE not found: {RCB_SOURCE}")
        print("Edit RCB_SOURCE in this script to point to your ResearchClawBench repo.")
        raise SystemExit(1)
    export_tasks()
    export_runs()
    export_leaderboard()
    copy_static()
    print("Done!")
