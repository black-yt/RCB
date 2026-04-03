import yaml
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path('.')
DATA_YAML = BASE / 'data' / '2111.01152' / '2111.01152.yaml'
OUTPUT_DIR = BASE / 'outputs'
FIG_DIR = BASE / 'report' / 'images'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_YAML, 'r') as f:
    entries = yaml.safe_load(f)

# filter to task entries (skip branch-only header)
tasks = [e for e in entries if 'task' in e]

for t in tasks:
    score = t.get('score', {}) or {}
    t['score_total'] = sum(score.get(k, 0) for k in ['in_paper','prompt_quality','follow_instructions','physics_logic','math_derivation','final_answer_accuracy'])

with open(OUTPUT_DIR / 'tasks_with_scores.json', 'w') as f:
    json.dump(tasks, f, indent=2)

per_task_scores = {t['task']: t['score_total'] for t in tasks}

plt.figure(figsize=(10,6))
items = list(per_task_scores.items())
labels = [k for k,_ in items]
values = [v for _,v in items]
order = sorted(range(len(values)), key=lambda i: values[i], reverse=True)
labels = [labels[i] for i in order]
values = [values[i] for i in order]

sns.barplot(x=values, y=labels, orient='h', palette='viridis')
plt.xlabel('Total Step Score')
plt.ylabel('Task')
plt.tight_layout()
plt.savefig(FIG_DIR / 'step_scores.png', dpi=300)
plt.close()

components = ['in_paper','prompt_quality','follow_instructions','physics_logic','math_derivation','final_answer_accuracy']

comp_avgs = {c: sum((t.get('score') or {}).get(c,0) for t in tasks)/len(tasks) for c in components}

plt.figure(figsize=(8,5))
plt.bar(list(comp_avgs.keys()), list(comp_avgs.values()), color='tab:blue')
plt.ylabel('Average Score')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(FIG_DIR / 'component_average_scores.png', dpi=300)
plt.close()

with open(OUTPUT_DIR / 'component_average_scores.json','w') as f:
    json.dump(comp_avgs, f, indent=2)
