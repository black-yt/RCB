import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml


ROOT = Path('.')
DATA_DIR = ROOT / 'data' / '2111.01152'
OUT_DIR = ROOT / 'outputs'
IMG_DIR = ROOT / 'report' / 'images'

OUT_DIR.mkdir(exist_ok=True, parents=True)
IMG_DIR.mkdir(exist_ok=True, parents=True)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 160


def read_text(path: Path) -> str:
    return path.read_text(encoding='utf-8')


def normalize_formula(text: str) -> str:
    text = text.replace('\n', ' ')
    text = re.sub(r'\\label\{[^}]+\}', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_reference_equations(tex_main: str, tex_sm: str):
    refs = {}

    main_h = re.search(r'\\begin\{equation\}\\label\{eq:Ham\}(.*?)\\end\{equation\}', tex_main, re.S)
    if main_h:
        refs['continuum_hamiltonian_main_text'] = normalize_formula(main_h.group(1))

    delta_b = re.search(r'\\begin\{equation\}\\label\{eq:Delta_b\}(.*?)\\end\{equation\}', tex_main, re.S)
    if delta_b:
        refs['bottom_layer_potential'] = normalize_formula(delta_b.group(1))

    delta_t = re.search(r'We set \\Delta_\{\\mathfrak\{t\}\}\(\\bm\{r\}\)=0', tex_main)
    refs['top_layer_potential'] = '\\Delta_{\\mathfrak{t}}(\\bm{r})=0' if delta_t else ''

    delta_T = re.search(r'\\begin\{equation\}\\label\{eq:Delta_T\}(.*?)\\end\{equation\}', tex_main, re.S)
    if delta_T:
        refs['interlayer_tunneling_main_text'] = normalize_formula(delta_T.group(1))

    sm_h0 = re.search(r'The single-particle Hamiltonian.*?\\begin\{eqnarray\}(.*?)\\end\{eqnarray\}', tex_sm, re.S)
    if sm_h0:
        refs['single_particle_second_quantized'] = normalize_formula(sm_h0.group(1))

    full_h = re.search(r'\\begin\{equation\}\\label\{eq:full\}(.*?)\\end\{equation\}', tex_sm, re.S)
    if full_h:
        refs['full_hamiltonian_hole_basis'] = normalize_formula(full_h.group(1))

    hf_h = re.search(r'\\begin\{equation\}\\label\{eq:HF\}(.*?)\\end\{equation\}', tex_sm, re.S)
    if hf_h:
        refs['hartree_fock_hamiltonian'] = normalize_formula(hf_h.group(1))

    return refs


def parse_task_yaml(path: Path):
    data = yaml.safe_load(read_text(path))
    records = []
    for entry in data:
        task = entry.get('task')
        answer = entry.get('answer', '')
        score = entry.get('score', {}) or {}
        placeholder = entry.get('placeholder', {}) or {}
        human_vs_llm = []
        for key, val in placeholder.items():
            if isinstance(val, dict):
                llm = val.get('LLM')
                human = val.get('human')
                llm_str = '' if llm is None else str(llm).strip()
                human_str = '' if human is None else str(human).strip()
                comparable = human_str not in {'', '(?)'}
                mismatch = (llm_str != human_str) if comparable else False
                human_vs_llm.append({
                    'task': task,
                    'placeholder': key,
                    'llm': llm,
                    'human': human,
                    'llm_str': llm_str,
                    'human_str': human_str,
                    'comparable': comparable,
                    'mismatch': mismatch,
                    'scores': val.get('score', {}),
                })
        rec = {
            'task': task,
            'answer': answer,
            **{k: score.get(k) for k in ['in_paper', 'prompt_quality', 'follow_instructions', 'physics_logic', 'math_derivation', 'final_answer_accuracy']}
        }
        records.append((rec, human_vs_llm))
    return records


def extract_auto_sections(auto_md: str):
    pattern = re.compile(r'##\s+(.*?)\n\*\*Prompt:\*\*\s*(.*?)\n\*\*Completion:\*\*\s*(.*?)(?=\n##\s+|\Z)', re.S)
    sections = []
    for match in pattern.finditer(auto_md):
        title, prompt, completion = match.groups()
        sections.append({
            'task': title.strip(),
            'prompt': prompt.strip(),
            'completion': completion.strip(),
        })
    return sections


def jaccard_tokens(a: str, b: str) -> float:
    tok = lambda s: set(re.findall(r'[A-Za-z_\\]+|\d+|[+\-*/^=]', s))
    sa, sb = tok(a), tok(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def compute_validation(auto_sections, refs):
    mapping = {
        'Construct Kinetic Hamiltonian (continuum version, single-particle)': 'continuum_hamiltonian_main_text',
        'Define each term in Kinetic Hamiltonian (continuum version)': 'continuum_hamiltonian_main_text',
        'Construct Potential Hamiltonian (continuum version)': 'continuum_hamiltonian_main_text',
        'Define each term in Potential Hamiltonian (continuum version)': 'interlayer_tunneling_main_text',
        'Convert from single-particle to second-quantized form, return in matrix': 'single_particle_second_quantized',
        'Convert from single-particle to second-quantized form, return in summation (expand the matrix)': 'single_particle_second_quantized',
    }
    rows = []
    for sec in auto_sections:
        ref_key = mapping.get(sec['task'])
        ref = refs.get(ref_key, '')
        sim = jaccard_tokens(sec['completion'], ref) if ref else math.nan
        rows.append({
            'task': sec['task'],
            'reference_key': ref_key,
            'token_jaccard': sim,
            'completion_length': len(sec['completion']),
            'reference_length': len(ref),
        })
    return rows


def save_fig_score_heatmap(score_df: pd.DataFrame):
    metrics = ['in_paper', 'prompt_quality', 'follow_instructions', 'physics_logic', 'math_derivation', 'final_answer_accuracy']
    plot_df = score_df.set_index('task')[metrics]
    fig, ax = plt.subplots(figsize=(14, 7))
    sns.heatmap(plot_df, annot=True, fmt='.0f', cmap='YlGnBu', vmin=0, vmax=2, cbar_kws={'label': 'Manual score'}, ax=ax)
    ax.set_title('Step-level manual scores for extracted tasks')
    fig.subplots_adjust(left=0.33, right=0.97, top=0.90, bottom=0.12)
    out = IMG_DIR / 'score_heatmap.png'
    fig.savefig(out)
    plt.close(fig)


def save_fig_placeholder_mismatch(placeholder_df: pd.DataFrame):
    comp_df = placeholder_df[placeholder_df['comparable']].copy()
    mismatch_counts = comp_df.groupby('task')['mismatch'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(x=mismatch_counts.index, y=mismatch_counts.values, color='#4C72B0', ax=ax)
    ax.tick_params(axis='x', rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    ax.set_ylabel('Mismatch rate')
    ax.set_xlabel('Task')
    ax.set_ylim(0, 1)
    ax.set_title('Placeholder mismatch rate on comparable fields only')
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.35)
    out = IMG_DIR / 'placeholder_mismatch_rate.png'
    fig.savefig(out)
    plt.close(fig)


def save_fig_validation(validation_df: pd.DataFrame):
    plot_df = validation_df.dropna(subset=['token_jaccard']).sort_values('token_jaccard', ascending=False)
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.barplot(data=plot_df, x='task', y='token_jaccard', color='#55A868', ax=ax)
    ax.tick_params(axis='x', rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    ax.set_ylabel('Token Jaccard similarity')
    ax.set_xlabel('Task')
    ax.set_ylim(0, 1)
    ax.set_title('Similarity between auto-completions and reference source equations')
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.35)
    out = IMG_DIR / 'completion_reference_similarity.png'
    fig.savefig(out)
    plt.close(fig)


def save_fig_score_distribution(score_df: pd.DataFrame):
    metrics = ['in_paper', 'prompt_quality', 'follow_instructions', 'physics_logic', 'math_derivation', 'final_answer_accuracy']
    long_df = score_df.melt(id_vars=['task'], value_vars=metrics, var_name='metric', value_name='score')
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=long_df, x='metric', y='score', color='#C44E52', ax=ax)
    ax.tick_params(axis='x', rotation=25)
    for label in ax.get_xticklabels():
        label.set_ha('right')
    ax.set_ylim(-0.1, 2.1)
    ax.set_title('Distribution of rubric scores across evaluation dimensions')
    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.28)
    out = IMG_DIR / 'score_distribution.png'
    fig.savefig(out)
    plt.close(fig)


def main():
    tex_main = read_text(DATA_DIR / '2111.01152.tex')
    tex_sm = read_text(DATA_DIR / '2111.01152_SM.tex')
    auto_md = read_text(DATA_DIR / '2111.01152_auto.md')

    refs = extract_reference_equations(tex_main, tex_sm)
    (OUT_DIR / 'reference_equations.json').write_text(json.dumps(refs, indent=2), encoding='utf-8')

    parsed_yaml = parse_task_yaml(DATA_DIR / '2111.01152.yaml')
    score_rows = [x[0] for x in parsed_yaml]
    placeholder_rows = [item for _, sub in parsed_yaml for item in sub]

    score_df = pd.DataFrame(score_rows)
    placeholder_df = pd.DataFrame(placeholder_rows)

    score_df.to_csv(OUT_DIR / 'step_scores.csv', index=False)
    placeholder_df.to_csv(OUT_DIR / 'placeholder_analysis.csv', index=False)

    auto_sections = extract_auto_sections(auto_md)
    (OUT_DIR / 'auto_sections.json').write_text(json.dumps(auto_sections, indent=2), encoding='utf-8')

    validation_rows = compute_validation(auto_sections, refs)
    validation_df = pd.DataFrame(validation_rows)
    validation_df.to_csv(OUT_DIR / 'completion_validation.csv', index=False)

    task_summary = {
        'paper_id': '2111.01152',
        'num_tasks_scored': int(len(score_df)),
        'num_placeholders': int(len(placeholder_df)),
        'num_auto_sections': int(len(auto_sections)),
        'mean_scores': score_df[[c for c in score_df.columns if c not in ['task', 'answer']]].mean(numeric_only=True).round(3).to_dict(),
        'placeholder_mismatch_rate': float(placeholder_df.loc[placeholder_df['comparable'], 'mismatch'].mean()) if placeholder_df['comparable'].any() else None,
        'mean_completion_reference_similarity': float(validation_df['token_jaccard'].dropna().mean()) if validation_df['token_jaccard'].notna().any() else None,
    }
    (OUT_DIR / 'task_summary.json').write_text(json.dumps(task_summary, indent=2), encoding='utf-8')

    score_means_by_task = score_df[['task', 'in_paper', 'prompt_quality', 'follow_instructions', 'physics_logic', 'math_derivation', 'final_answer_accuracy']].copy()
    score_means_by_task['average_score'] = score_means_by_task.drop(columns=['task']).mean(axis=1)

    mismatch_by_task = placeholder_df.groupby('task').agg(
        total_placeholders=('placeholder', 'count'),
        comparable_placeholders=('comparable', 'sum'),
        mismatches=('mismatch', 'sum'),
    ).reset_index()
    mismatch_by_task['mismatch_rate'] = mismatch_by_task.apply(
        lambda row: (row['mismatches'] / row['comparable_placeholders']) if row['comparable_placeholders'] else np.nan,
        axis=1,
    )

    validation_join = validation_df[['task', 'token_jaccard']]
    merged = score_means_by_task.merge(mismatch_by_task, on='task', how='left').merge(validation_join, on='task', how='left')
    validation_report = {
        'task_level_summary': merged.to_dict(orient='records'),
        'lowest_follow_instruction_tasks': score_df.nsmallest(3, 'follow_instructions')[['task', 'follow_instructions']].to_dict(orient='records'),
        'highest_mismatch_tasks': mismatch_by_task.sort_values('mismatch_rate', ascending=False).head(3).to_dict(orient='records'),
    }
    (OUT_DIR / 'validation_report.json').write_text(json.dumps(validation_report, indent=2), encoding='utf-8')

    save_fig_score_heatmap(score_df)
    save_fig_placeholder_mismatch(placeholder_df)
    save_fig_validation(validation_df)
    save_fig_score_distribution(score_df)

    print(json.dumps(task_summary, indent=2))


if __name__ == '__main__':
    main()
