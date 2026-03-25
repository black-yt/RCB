"""Export evaluation data from ResearchClawBench to static JSON for GitHub Pages.

Standalone script — no imports from ResearchClawBench.
Configure RCB_SOURCE below to point to your ResearchClawBench repo root.
"""

import json
import re
import shutil
import sys
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
# Load INSTRUCTIONS_TEMPLATE dynamically from RCB source (always up to date)
# ---------------------------------------------------------------------------

def _load_instructions_template():
    """Load INSTRUCTIONS_TEMPLATE from ResearchClawBench/evaluation/instructions_tmpl.py."""
    tmpl_path = RCB_SOURCE / "evaluation" / "instructions_tmpl.py"
    if not tmpl_path.exists():
        raise FileNotFoundError(f"instructions_tmpl.py not found: {tmpl_path}")
    ns = {}
    exec(tmpl_path.read_text(encoding="utf-8"), ns)
    return ns["INSTRUCTIONS_TEMPLATE"]


INSTRUCTIONS_TEMPLATE = _load_instructions_template()


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


def _build_file_tree(root, prefix="", max_per_dir=0, max_depth=0):
    """Build flat file tree list for a directory."""
    skip_names = {
        "_meta.json", "_agent_output.jsonl", "_score.json",
        ".claude", "__pycache__",
        # nanobot internal files
        "AGENTS.md", "HEARTBEAT.md", "SOUL.md", "TOOLS.md", "USER.md",
        "sessions", "memory", "skills",
    }
    tree = []

    def _walk(root, prefix, depth):
        try:
            entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return
        entries = [e for e in entries if not e.name.startswith(".") and e.name not in skip_names]
        total = len(entries)
        limited = max_per_dir and total > max_per_dir
        if limited:
            entries = entries[:max_per_dir]
        for entry in entries:
            rel = f"{prefix}/{entry.name}" if prefix else entry.name
            if entry.is_dir():
                node = {"name": entry.name, "path": rel, "type": "directory"}
                tree.append(node)
                if max_depth and depth >= max_depth:
                    node["truncated"] = True
                else:
                    _walk(entry, rel, depth + 1)
            else:
                try:
                    stat = entry.stat()
                except OSError:
                    continue
                tree.append({
                    "name": entry.name, "path": rel, "type": "file",
                    "size": stat.st_size, "mtime": stat.st_mtime,
                })
        if limited:
            tree.append({"name": f"… {total - max_per_dir} more items", "path": prefix + "/_more", "type": "truncated"})

    _walk(root, prefix, 1)
    return tree


def _build_instructions(task_info, workspace="<workspace>"):
    """Build INSTRUCTIONS.md content using the live template from ResearchClawBench."""
    task_desc = task_info.get("task", "")
    data_parts = []
    for d in task_info.get("data", []):
        ws_path = d.get("path", "").lstrip("./")
        data_type = d.get("type", "")
        type_str = f" [{data_type}]" if data_type else ""
        data_parts.append(f"- **{d['name']}**{type_str} (`{ws_path}`): {d.get('description', '')}")
    data_text = "\n".join(data_parts) if data_parts else "No specific data files."
    return INSTRUCTIONS_TEMPLATE.format(
        workspace=workspace,
        task_desc=task_desc,
        data_text=data_text,
    )


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
                    top_dirs[subdir] = _build_file_tree(sub_path, subdir, max_per_dir=20, max_depth=4)
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
    skipped = 0
    for run in runs:
        ws = _get_run_workspace(run["run_id"])
        if not ws:
            continue
        meta_path = ws / "_meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Strict 3-condition filter: completed + report exists + score exists
        if meta.get("status") != "completed":
            skipped += 1
            continue
        report_path = ws / "report" / "report.md"
        if not report_path.exists():
            skipped += 1
            continue
        score_path = ws / "_score.json"
        if not score_path.exists():
            skipped += 1
            continue
        try:
            with open(score_path, "r", encoding="utf-8") as f:
                score_data = json.load(f)
            if "total_score" not in score_data:
                skipped += 1
                continue
        except (json.JSONDecodeError, OSError):
            skipped += 1
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
            "score": score_data,
            "report": report_path.read_text(encoding="utf-8", errors="replace"),
        }

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

        # File tree — only export known directories + INSTRUCTIONS.md
        EXPORT_DIRS = ["code", "data", "outputs", "related_work", "report"]
        INPUT_DIRS = {"data", "related_work"}
        NO_LIMIT_DIRS = {"report"}
        tree = []
        for subdir in EXPORT_DIRS:
            sub = ws / subdir
            if sub.exists():
                tree.append({"name": subdir, "path": subdir, "type": "directory"})
                if subdir in NO_LIMIT_DIRS:
                    tree.extend(_build_file_tree(sub, subdir))
                else:
                    depth = 4 if subdir in INPUT_DIRS else 3
                    tree.extend(_build_file_tree(sub, subdir, max_per_dir=20, max_depth=depth))
        instr = ws / "INSTRUCTIONS.md"
        if instr.exists():
            st = instr.stat()
            tree.append({"name": "INSTRUCTIONS.md", "path": "INSTRUCTIONS.md", "type": "file", "size": st.st_size, "mtime": st.st_mtime})
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

    print(f"Exported {len(exported)} runs (skipped {skipped})")


def export_leaderboard():
    runs = _list_runs()
    best = {}
    for run in runs:
        ws = _get_run_workspace(run["run_id"])
        if not ws:
            continue
        # Require all 3 conditions: completed + report + score
        meta_path = ws / "_meta.json"
        if not meta_path.exists():
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if meta.get("status") != "completed":
            continue
        if not (ws / "report" / "report.md").exists():
            continue
        score_path = ws / "_score.json"
        if not score_path.exists():
            continue
        try:
            with open(score_path, "r", encoding="utf-8") as f:
                score_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        if "total_score" not in score_data:
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
    """Sync frontend assets from ResearchClawBench."""
    dst = HOME_DIR / "static"
    # logos and favicon
    for d in ["logos"]:
        (dst / d).mkdir(parents=True, exist_ok=True)
        for f in (STATIC_SRC / d).iterdir():
            shutil.copy2(f, dst / d / f.name)
    shutil.copy2(STATIC_SRC / "favicon.svg", dst / "favicon.svg")
    # frontend JS and CSS (must stay in sync)
    shutil.copy2(STATIC_SRC / "app.js", dst / "app.js")
    shutil.copy2(STATIC_SRC / "style.css", dst / "style.css")
    print("Copied static assets (logos, favicon, app.js, style.css)")


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
