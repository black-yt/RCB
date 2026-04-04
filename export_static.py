"""Export evaluation data from ResearchClawBench to static JSON for GitHub Pages.

Standalone script — no imports from ResearchClawBench.
Configure RCB_SOURCE below to point to your ResearchClawBench repo root.
"""

import json
import hashlib
import os
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — adjust RCB_SOURCE if repos are not siblings
# ---------------------------------------------------------------------------
HOME_DIR = Path(__file__).resolve().parent
DATA_DIR = HOME_DIR / "data"
EXPORT_STATE_DIR = HOME_DIR / "export_state"

RCB_SOURCE = HOME_DIR.parent / "ResearchClawBench"
TASKS_DIR = RCB_SOURCE / "tasks"
WORKSPACES_DIR = RCB_SOURCE / "workspaces"
STATIC_SRC = RCB_SOURCE / "evaluation" / "static"

# Viewable text extensions (must match app.js)
TEXT_EXTS = {
    '.txt', '.md', '.py', '.js', '.json', '.jsonl', '.csv', '.tsv',
    '.yml', '.yaml', '.sh', '.bash', '.r', '.R', '.html', '.css', '.xml',
    '.ini', '.cfg', '.conf', '.toml', '.log', '.dat', '.tex', '.bib',
    '.sql', '.c', '.cpp', '.h', '.java', '.go', '.rs', '.jl', '.m', '.ipynb',
}
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg'}

MODEL_ESTIMATED_USD_PER_MIN = {
    "gpt-5.4": 0.15,
    "claude-sonnet-4-6": 0.20,
    "claude-opus-4-6": 0.40,
}
# Bump when task export output/schema changes and existing cached signatures must be invalidated.
TASK_EXPORT_VERSION = 4
# Bump when run export output/schema changes and existing cached signatures must be invalidated.
RUN_EXPORT_VERSION = 3
TASK_EXPORT_MANIFEST = EXPORT_STATE_DIR / "task_export_manifest.json"
RUN_EXPORT_MANIFEST = EXPORT_STATE_DIR / "run_export_manifest.json"
RUN_OUTPUT_FILES = ["_agent_output.jsonl", "_claude_output.jsonl"]
RUN_EXPORT_DIRS = ["code", "data", "outputs", "related_work", "report"]
RUN_SIGNATURE_DIRS = ["code", "outputs", "report"]
RUN_INPUT_DIRS = {"data", "related_work"}
RUN_NO_LIMIT_DIRS = {"report"}
RUN_MAX_PER_DIR = 10
RUN_MAX_DEPTH = 3
RUN_SIGNATURE_SKIP_NAMES = {
    ".claude", "__pycache__",
    "AGENTS.md", "HEARTBEAT.md", "SOUL.md", "TOOLS.md", "USER.md",
    "sessions", "memory", "skills",
}

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


def _list_tasks_grouped(task_ids=None):
    """Return {domain: [task_id, ...]}."""
    groups = {}
    for task_id in (task_ids if task_ids is not None else _list_tasks()):
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
        model = _normalize_model_name(meta.get("model", ""))
        runs.append({
            "run_id": d.name,
            "task_id": meta.get("task_id"),
            "timestamp": meta.get("timestamp"),
            "status": meta.get("status", "unknown"),
            "agent_name": meta.get("agent_name", ""),
            "model": model,
            "model_display": _format_model_display(model),
            "duration_seconds": meta.get("duration_seconds"),
            "workspace": str(d),
        })
    return runs


def _get_run_workspace(run_id):
    ws = WORKSPACES_DIR / run_id
    return ws if ws.is_dir() else None


def _normalize_model_name(model):
    raw = (model or "").strip()
    raw = re.sub(r"\x1b\[[0-9;]*m", "", raw)
    raw = re.sub(r"\[[^\]]+\]$", "", raw)
    if raw.startswith("openai/"):
        raw = raw.split("/", 1)[1]
    return raw


def _normalize_pricing_model(model):
    raw = _normalize_model_name(model)
    if raw in MODEL_ESTIMATED_USD_PER_MIN:
        return raw
    if raw.startswith("claude-sonnet-4-6"):
        return "claude-sonnet-4-6"
    if raw.startswith("claude-opus-4-6"):
        return "claude-opus-4-6"
    return raw


def _format_model_display(model):
    normalized = _normalize_model_name(model)
    if normalized == "gpt-5.4":
        return "GPT-5.4"
    if normalized.startswith("claude-opus-4-6"):
        return "Opus 4.6"
    if normalized.startswith("claude-sonnet-4-6"):
        return "Sonnet 4.6"
    return normalized


def _estimate_run_cost_usd(model, duration_seconds):
    pricing_model = _normalize_pricing_model(model)
    rate_per_min = MODEL_ESTIMATED_USD_PER_MIN.get(pricing_model)
    if not rate_per_min or duration_seconds is None:
        return None
    duration_minutes = max(float(duration_seconds), 0.0) / 60.0
    return round(duration_minutes * rate_per_min, 6)


def _load_export_manifest(path, version):
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict) or data.get("version") != version:
        return {}
    runs = data.get("runs", {})
    return runs if isinstance(runs, dict) else {}


def _save_export_manifest(path, version, runs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"version": version, "runs": runs}, f, indent=2, sort_keys=True)


def _find_run_output_path(ws):
    for output_name in RUN_OUTPUT_FILES:
        output_path = ws / output_name
        if output_path.exists():
            return output_path
    return None


def _update_signature_file(hasher, path, rel_path):
    try:
        st = path.stat()
    except OSError:
        return
    hasher.update(f"F|{rel_path}|{st.st_size}|{st.st_mtime_ns}\n".encode("utf-8"))


def _collect_tree_summary(root, skip_names=None):
    skip_names = skip_names or set()
    summary = {"dirs": 0, "files": 0, "size": 0, "max_mtime_ns": 0}
    if not root.exists():
        summary["path_hash"] = "0"
        return summary

    path_hasher = hashlib.sha256()

    def _walk(current, rel_prefix=""):
        try:
            entries = list(os.scandir(current))
        except OSError:
            return
        entries = sorted(
            (entry for entry in entries if not entry.name.startswith(".") and entry.name not in skip_names),
            key=lambda entry: entry.name.lower(),
        )
        for entry in entries:
            rel_path = f"{rel_prefix}/{entry.name}" if rel_prefix else entry.name
            if entry.name.startswith(".") or entry.name in skip_names:
                continue
            try:
                st = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            summary["max_mtime_ns"] = max(summary["max_mtime_ns"], st.st_mtime_ns)
            if entry.is_dir(follow_symlinks=False):
                summary["dirs"] += 1
                path_hasher.update(f"D|{rel_path}\n".encode("utf-8"))
                _walk(entry.path, rel_path)
            else:
                summary["files"] += 1
                summary["size"] += st.st_size
                path_hasher.update(f"F|{rel_path}\n".encode("utf-8"))

    _walk(root)
    summary["path_hash"] = path_hasher.hexdigest()
    return summary


def _update_signature_summary(hasher, label, root, skip_names=None):
    summary = _collect_tree_summary(root, skip_names=skip_names)
    hasher.update(
        f"S|{label}|{summary['dirs']}|{summary['files']}|{summary['size']}|{summary['max_mtime_ns']}|{summary['path_hash']}\n".encode("utf-8")
    )


def _compute_run_export_signature(ws):
    hasher = hashlib.sha256()
    hasher.update(f"run-export-v{RUN_EXPORT_VERSION}\n".encode("utf-8"))
    for rel_path in ["_meta.json", "_score.json", "INSTRUCTIONS.md", "report/report.md"]:
        _update_signature_file(hasher, ws / rel_path, rel_path)
    output_path = _find_run_output_path(ws)
    if output_path:
        _update_signature_file(hasher, output_path, output_path.name)
    for subdir in RUN_SIGNATURE_DIRS:
        _update_signature_summary(hasher, subdir, ws / subdir, skip_names=RUN_SIGNATURE_SKIP_NAMES)
    return hasher.hexdigest()


def _run_export_complete(run_out_dir, has_output):
    if not run_out_dir.exists():
        return False
    required = [run_out_dir / "data.json", run_out_dir / "files.json"]
    if has_output:
        required.append(run_out_dir / "output.json")
    if not all(path.exists() for path in required):
        return False
    return _exported_workspace_files_present(run_out_dir)


def _get_task_export_paper_path(src_task):
    for pdf in sorted((src_task / "target_study").glob("paper*.pdf")):
        if pdf.stat().st_size < 5 * 1024 * 1024:
            return pdf
    return None


def _build_task_export_state(task_id):
    src_task = TASKS_DIR / task_id
    info = _load_task_info(task_id)
    try:
        checklist = _load_checklist(task_id)
    except FileNotFoundError:
        checklist = None
    instructions_text = _build_instructions(info)

    tree = []
    top_dirs = {}
    for subdir in ["data", "related_work"]:
        sub_path = src_task / subdir
        if sub_path.exists():
            top_dirs[subdir] = _build_file_tree(sub_path, subdir, max_per_dir=10, max_depth=3)
    for d in ["code", "outputs", "report"]:
        if d not in top_dirs:
            top_dirs[d] = []
    top_dirs.setdefault("report", []).insert(0, {"name": "images", "path": "report/images", "type": "directory"})
    for name in sorted(top_dirs.keys()):
        tree.append({"name": name, "path": name, "type": "directory"})
        tree.extend(top_dirs[name])
    tree.append({"name": "INSTRUCTIONS.md", "path": "INSTRUCTIONS.md", "type": "file", "size": len(instructions_text)})
    return src_task, info, checklist, instructions_text, tree


def _compute_task_export_signature(task_id):
    hasher = hashlib.sha256()
    hasher.update(f"task-export-v{TASK_EXPORT_VERSION}\n".encode("utf-8"))
    src_task = TASKS_DIR / task_id
    _update_signature_file(hasher, src_task / "task_info.json", "task_info.json")
    _update_signature_file(hasher, src_task / "target_study" / "checklist.json", "target_study/checklist.json")
    _update_signature_file(hasher, RCB_SOURCE / "evaluation" / "instructions_tmpl.py", "evaluation/instructions_tmpl.py")
    paper = _get_task_export_paper_path(src_task)
    if paper:
        _update_signature_file(hasher, paper, f"target_study/{paper.name}")
    for subdir in ["data", "related_work"]:
        _update_signature_summary(hasher, subdir, src_task / subdir)
    _update_signature_summary(hasher, "target_study/images", src_task / "target_study" / "images")
    return hasher.hexdigest()


def _task_export_complete(task_id, task_dir):
    if not task_dir.exists():
        return False
    required = [
        task_dir / "info.json",
        task_dir / "files.json",
        task_dir / "INSTRUCTIONS.md",
        task_dir / "workspace" / "INSTRUCTIONS.md",
    ]
    src_task = TASKS_DIR / task_id
    if (src_task / "target_study" / "checklist.json").exists():
        required.append(task_dir / "checklist.json")
    if _get_task_export_paper_path(src_task):
        required.append(task_dir / "paper.pdf")
    src_images_dir = src_task / "target_study" / "images"
    if src_images_dir.exists():
        required.append(task_dir / "images")
    if not all(path.exists() for path in required):
        return False
    return _exported_workspace_files_present(task_dir)


def _exported_workspace_files_present(base_dir):
    files_json = base_dir / "files.json"
    try:
        with open(files_json, "r", encoding="utf-8") as f:
            items = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    if not isinstance(items, list):
        return False

    workspace_dir = base_dir / "workspace"
    for item in items:
        if not isinstance(item, dict) or item.get("type") != "file":
            continue
        if item.get("shared"):
            continue
        if not item.get("exported"):
            continue
        rel_path = item.get("path")
        if not isinstance(rel_path, str) or not rel_path:
            return False
        if not (workspace_dir / rel_path).exists():
            return False
    return True


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

def export_tasks(task_ids=None):
    task_ids = list(task_ids if task_ids is not None else _list_tasks())
    grouped = _list_tasks_grouped(task_ids)
    tasks_root = DATA_DIR / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    prev_manifest = _load_export_manifest(TASK_EXPORT_MANIFEST, TASK_EXPORT_VERSION)
    next_manifest = {}
    refreshed = 0
    reused = 0
    removed = 0
    valid_task_ids = set()
    with open(DATA_DIR / "tasks.json", "w", encoding="utf-8") as f:
        json.dump(grouped, f, indent=2)

    for domain, task_ids in grouped.items():
        for task_id in task_ids:
            valid_task_ids.add(task_id)
            task_dir = DATA_DIR / "tasks" / task_id
            signature = _compute_task_export_signature(task_id)
            next_manifest[task_id] = signature
            if prev_manifest.get(task_id) == signature and _task_export_complete(task_id, task_dir):
                reused += 1
                continue
            if task_dir.exists():
                shutil.rmtree(task_dir)
            task_dir.mkdir(parents=True, exist_ok=True)
            src_task, info, checklist, instructions_text, tree = _build_task_export_state(task_id)
            with open(task_dir / "info.json", "w", encoding="utf-8") as f:
                json.dump(info, f, indent=2)
            if checklist is not None:
                with open(task_dir / "checklist.json", "w", encoding="utf-8") as f:
                    json.dump(checklist, f, indent=2)
            # Copy checklist images
            images_dir = src_task / "target_study" / "images"
            if images_dir.exists():
                dst = task_dir / "images"
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(images_dir, dst)

            # Copy target paper PDF (skip if > 5MB)
            paper = _get_task_export_paper_path(src_task)
            if paper:
                shutil.copy2(paper, task_dir / "paper.pdf")

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
                max_size = 5 * 1024 * 1024 if ext == '.pdf' else 1 * 1024 * 1024
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
            refreshed += 1

    for task_dir in tasks_root.iterdir():
        if not task_dir.is_dir() or task_dir.name in valid_task_ids:
            continue
        shutil.rmtree(task_dir)
        removed += 1

    _save_export_manifest(TASK_EXPORT_MANIFEST, TASK_EXPORT_VERSION, next_manifest)
    print(f"Exported {refreshed} tasks, reused {reused}, removed {removed}")


def export_runs(runs=None):
    runs = list(runs if runs is not None else _list_runs())
    runs_dir = DATA_DIR / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    prev_manifest = _load_export_manifest(RUN_EXPORT_MANIFEST, RUN_EXPORT_VERSION)
    next_manifest = {}

    index = []
    skipped = 0
    refreshed = 0
    reused = 0
    valid_run_ids = set()
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

        valid_run_ids.add(run["run_id"])
        run_out_dir = runs_dir / run["run_id"]
        output_path = _find_run_output_path(ws)
        signature = _compute_run_export_signature(ws)
        next_manifest[run["run_id"]] = signature
        if prev_manifest.get(run["run_id"]) == signature and _run_export_complete(run_out_dir, has_output=bool(output_path)):
            reused += 1
        else:
            if run_out_dir.exists():
                shutil.rmtree(run_out_dir)
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
            if output_path:
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

            # File tree — agent output dirs copied; input dirs shown as shared (loaded from task workspace)
            tree = []
            for subdir in RUN_EXPORT_DIRS:
                sub = ws / subdir
                if sub.exists():
                    tree.append({"name": subdir, "path": subdir, "type": "directory"})
                    if subdir in RUN_NO_LIMIT_DIRS:
                        tree.extend(_build_file_tree(sub, subdir))
                    else:
                        tree.extend(_build_file_tree(sub, subdir, max_per_dir=RUN_MAX_PER_DIR, max_depth=RUN_MAX_DEPTH))
            instr = ws / "INSTRUCTIONS.md"
            if instr.exists():
                st = instr.stat()
                tree.append({"name": "INSTRUCTIONS.md", "path": "INSTRUCTIONS.md", "type": "file", "size": st.st_size, "mtime": st.st_mtime})

            # Mark input files as shared (served from task workspace, not copied per-run)
            for item in tree:
                if item["type"] == "file":
                    top = item["path"].split("/")[0]
                    if top in RUN_INPUT_DIRS:
                        item["shared"] = True  # frontend loads from data/tasks/{task_id}/workspace/

            with open(run_out_dir / "files.json", "w", encoding="utf-8") as f:
                json.dump(tree, f, indent=2)

            # Copy viewable files — skip input dirs (shared from task workspace)
            files_dst = run_out_dir / "workspace"
            if files_dst.exists():
                shutil.rmtree(files_dst)
            run_exported = set()
            for item in tree:
                if item["type"] != "file":
                    continue
                if item.get("shared"):
                    continue  # don't copy — served from task workspace
                src = ws / item["path"]
                if not src.exists():
                    continue
                ext = src.suffix.lower()
                if ext in TEXT_EXTS or ext in IMG_EXTS or ext == '.pdf':
                    max_size = 5 * 1024 * 1024 if ext == '.pdf' else 1 * 1024 * 1024
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
            refreshed += 1

        index.append({
            "run_id": run["run_id"],
            "task_id": meta.get("task_id"),
            "timestamp": meta.get("timestamp"),
            "status": meta.get("status"),
            "agent_name": meta.get("agent_name", ""),
            "model": run.get("model", ""),
            "model_display": run.get("model_display", ""),
            "duration_seconds": meta.get("duration_seconds"),
            "total_score": score_data.get("total_score"),
        })

    removed = 0
    for run_out_dir in runs_dir.iterdir():
        if not run_out_dir.is_dir() or run_out_dir.name in valid_run_ids:
            continue
        shutil.rmtree(run_out_dir)
        removed += 1

    _save_export_manifest(RUN_EXPORT_MANIFEST, RUN_EXPORT_VERSION, next_manifest)

    # runs_index.json
    with open(DATA_DIR / "runs_index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print(f"Exported {refreshed} runs, reused {reused}, removed {removed} (skipped {skipped})")


def export_leaderboard():
    runs_index_path = DATA_DIR / "runs_index.json"
    if not runs_index_path.exists():
        raise FileNotFoundError(f"runs_index.json not found: {runs_index_path}")
    with open(runs_index_path, "r", encoding="utf-8") as f:
        runs = json.load(f)
    best = {}
    for run in runs:
        if not isinstance(run, dict):
            continue
        total = run.get("total_score")
        if total is None:
            continue
        task_id = run["task_id"]
        agent = run.get("agent_name", "Unknown")
        cost_usd = _estimate_run_cost_usd(run.get("model", ""), run.get("duration_seconds"))
        entry = {
            "score": total,
            "run_id": run["run_id"],
            "duration_seconds": run.get("duration_seconds"),
            "cost_usd": cost_usd,
            "model": run.get("model", ""),
            "model_display": run.get("model_display", ""),
        }
        key = (task_id, agent)
        if key not in best or total > best[key]["score"]:
            best[key] = entry

    tasks_set, agents_set = set(), set()
    for (t, a) in best:
        tasks_set.add(t)
        agents_set.add(a)
    tasks_list, agents_list = sorted(tasks_set), sorted(agents_set)
    scores = {a: {t: best[(t, a)] for t in tasks_list if (t, a) in best} for a in agents_list}
    frontier = {}
    for task in tasks_list:
        best_entry = None
        for agent in agents_list:
            key = (task, agent)
            if key in best and (best_entry is None or best[key]["score"] > best_entry["score"]):
                best_entry = best[key]
        frontier[task] = best_entry["score"] if best_entry else None

    with open(DATA_DIR / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "tasks": tasks_list,
                "agents": agents_list,
                "scores": scores,
                "frontier": frontier,
            },
            f,
            indent=2,
        )
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
    task_ids = _list_tasks()
    runs = _list_runs()
    export_tasks(task_ids=task_ids)
    export_runs(runs=runs)
    export_leaderboard()
    copy_static()
    print("Done!")
