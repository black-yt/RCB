import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path('.')
DATA_CSV = ROOT / 'data' / 'dataset1_cloud_seeding_records' / 'cloud_seeding_us_2000_2025.csv'
STATE_GEOJSON = ROOT / 'data' / 'dataset1_cloud_seeding_records' / 'us_states.geojson'
OUTPUT_DIR = ROOT / 'outputs'
FIG_DIR = ROOT / 'report' / 'images'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200


def split_multi(series: pd.Series) -> pd.Series:
    parts = (
        series.fillna('')
        .astype(str)
        .str.lower()
        .str.replace('&', ',', regex=False)
        .str.split(',')
        .explode()
        .str.strip()
    )
    parts = parts[(parts != '') & (parts != 'nan')]
    return parts


def normalize_text(s: pd.Series) -> pd.Series:
    return s.fillna('unknown').astype(str).str.strip().str.lower()


def titlecase_words(text: str) -> str:
    return ' '.join(w.capitalize() for w in str(text).split())


def save_table(df: pd.DataFrame, name: str):
    df.to_csv(OUTPUT_DIR / name, index=False)


def main():
    df = pd.read_csv(DATA_CSV)
    df['year'] = pd.to_numeric(df['year'], errors='coerce').astype('Int64')
    df['state'] = normalize_text(df['state'])
    df['season'] = normalize_text(df['season'])
    df['operator_affiliation'] = normalize_text(df['operator_affiliation'])
    df['agent'] = normalize_text(df['agent'])
    df['apparatus'] = normalize_text(df['apparatus'])
    df['purpose'] = normalize_text(df['purpose'])

    overview = {
        'n_records': int(len(df)),
        'n_columns': int(df.shape[1]),
        'year_min': int(df['year'].min()),
        'year_max': int(df['year'].max()),
        'n_states': int(df['state'].nunique()),
        'n_projects': int(df['project'].nunique()),
        'n_operators': int(df['operator_affiliation'].nunique()),
    }
    (OUTPUT_DIR / 'data_overview.json').write_text(json.dumps(overview, indent=2))

    missingness = df.isna().sum().rename('missing_count').reset_index().rename(columns={'index': 'column'})
    missingness['missing_share'] = missingness['missing_count'] / len(df)
    save_table(missingness, 'missingness.csv')

    state_counts = (
        df.groupby('state', dropna=False)
        .size()
        .reset_index(name='records')
        .sort_values('records', ascending=False)
    )
    state_counts['share'] = state_counts['records'] / len(df)
    state_counts['state_title'] = state_counts['state'].map(titlecase_words)
    save_table(state_counts[['state', 'state_title', 'records', 'share']], 'state_counts.csv')

    year_counts = (
        df.groupby('year', dropna=False)
        .size()
        .reset_index(name='records')
        .sort_values('year')
    )
    year_counts['yoy_change'] = year_counts['records'].diff()
    year_counts['rolling_3yr_mean'] = year_counts['records'].rolling(3, min_periods=1).mean()
    save_table(year_counts, 'year_counts.csv')

    season_counts = split_multi(df['season']).value_counts().rename_axis('season').reset_index(name='records')
    season_counts['share'] = season_counts['records'] / season_counts['records'].sum()
    save_table(season_counts, 'season_counts_exploded.csv')

    purpose_counts = split_multi(df['purpose']).value_counts().rename_axis('purpose').reset_index(name='records')
    purpose_counts['share'] = purpose_counts['records'] / purpose_counts['records'].sum()
    save_table(purpose_counts, 'purpose_counts_exploded.csv')

    agent_counts = split_multi(df['agent']).value_counts().rename_axis('agent').reset_index(name='records')
    agent_counts['share'] = agent_counts['records'] / agent_counts['records'].sum()
    save_table(agent_counts, 'agent_counts_exploded.csv')

    apparatus_counts = split_multi(df['apparatus']).value_counts().rename_axis('apparatus').reset_index(name='records')
    apparatus_counts['share'] = apparatus_counts['records'] / apparatus_counts['records'].sum()
    save_table(apparatus_counts, 'apparatus_counts_exploded.csv')

    operator_counts = (
        df.groupby('operator_affiliation').size().reset_index(name='records').sort_values('records', ascending=False)
    )
    operator_counts['share'] = operator_counts['records'] / len(df)
    save_table(operator_counts, 'operator_counts.csv')

    state_year = df.groupby(['year', 'state']).size().reset_index(name='records')
    save_table(state_year, 'state_year_counts.csv')

    purpose_by_year = (
        df.assign(purpose_split=df['purpose'].str.split(','))
        .explode('purpose_split')
        .assign(purpose_split=lambda x: x['purpose_split'].str.strip())
        .groupby(['year', 'purpose_split']).size().reset_index(name='records')
        .sort_values(['year', 'records'], ascending=[True, False])
    )
    save_table(purpose_by_year, 'purpose_by_year.csv')

    ap = (
        df.assign(agent_split=df['agent'].str.split(','), apparatus_split=df['apparatus'].str.split(','))
        .explode('agent_split')
        .explode('apparatus_split')
    )
    ap['agent_split'] = ap['agent_split'].str.strip()
    ap['apparatus_split'] = ap['apparatus_split'].str.strip()
    ap = ap[(ap['agent_split'] != '') & (ap['apparatus_split'] != '')]
    ap_counts = ap.groupby(['agent_split', 'apparatus_split']).size().reset_index(name='records')
    save_table(ap_counts.sort_values('records', ascending=False), 'agent_apparatus_counts.csv')

    summary = {
        'top_states': state_counts[['state_title', 'records', 'share']].head(10).to_dict(orient='records'),
        'top_purposes': purpose_counts.head(10).to_dict(orient='records'),
        'top_agents': agent_counts.head(10).to_dict(orient='records'),
        'top_apparatus': apparatus_counts.head(10).to_dict(orient='records'),
        'top_operators': operator_counts.head(10).to_dict(orient='records'),
        'annual_peak_year': int(year_counts.loc[year_counts['records'].idxmax(), 'year']),
        'annual_peak_records': int(year_counts['records'].max()),
    }
    (OUTPUT_DIR / 'summary_stats.json').write_text(json.dumps(summary, indent=2))

    # Figure 1: data overview
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    sns.barplot(data=state_counts.head(10), y='state_title', x='records', ax=axes[0, 0], palette='Blues_r')
    axes[0, 0].set_title('Top states by reported records')
    axes[0, 0].set_xlabel('Records')
    axes[0, 0].set_ylabel('State')

    sns.barplot(data=season_counts, x='season', y='records', ax=axes[0, 1], palette='Greens_d')
    axes[0, 1].set_title('Season tags (exploded)')
    axes[0, 1].set_xlabel('Season')
    axes[0, 1].set_ylabel('Exploded records')
    axes[0, 1].tick_params(axis='x', rotation=25)

    sns.barplot(data=operator_counts.head(10), y='operator_affiliation', x='records', ax=axes[1, 0], palette='Purples_r')
    axes[1, 0].set_title('Top operators')
    axes[1, 0].set_xlabel('Records')
    axes[1, 0].set_ylabel('Operator affiliation')

    sns.histplot(data=df, x='year', discrete=True, ax=axes[1, 1], color='#cc4c02')
    axes[1, 1].set_title('Record distribution by year')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Records')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'data_overview.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 2: annual activity dynamics
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(year_counts['year'], year_counts['records'], marker='o', linewidth=2.5, label='Annual records')
    ax.plot(year_counts['year'], year_counts['rolling_3yr_mean'], linewidth=2, linestyle='--', label='3-year rolling mean')
    peak = year_counts.loc[year_counts['records'].idxmax()]
    ax.scatter([peak['year']], [peak['records']], color='crimson', s=80, zorder=5)
    ax.annotate(f"Peak: {int(peak['year'])} ({int(peak['records'])})", (peak['year'], peak['records']),
                xytext=(10, 10), textcoords='offset points', color='crimson')
    ax.set_title('Annual cloud-seeding reporting activity')
    ax.set_xlabel('Year')
    ax.set_ylabel('Project records')
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'annual_activity.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 3: purpose composition
    fig, ax = plt.subplots(figsize=(12, 7))
    top_purposes = purpose_counts.head(8).copy()
    top_purposes['purpose_title'] = top_purposes['purpose'].map(titlecase_words)
    sns.barplot(data=top_purposes, y='purpose_title', x='records', ax=ax, palette='magma')
    ax.set_title('Purpose composition of reported activities')
    ax.set_xlabel('Exploded records')
    ax.set_ylabel('Purpose')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'purpose_composition.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 4: agent-apparatus deployment heatmap
    top_agent_names = agent_counts.head(6)['agent'].tolist()
    top_apparatus_names = apparatus_counts.head(5)['apparatus'].tolist()
    heat_df = ap_counts[ap_counts['agent_split'].isin(top_agent_names) & ap_counts['apparatus_split'].isin(top_apparatus_names)].copy()
    heat = heat_df.pivot(index='agent_split', columns='apparatus_split', values='records').fillna(0)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(heat, annot=True, fmt='.0f', cmap='YlGnBu', ax=ax)
    ax.set_title('Agent-apparatus deployment pattern')
    ax.set_xlabel('Apparatus')
    ax.set_ylabel('Agent')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'agent_apparatus_heatmap.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 5: state choropleth
    states = gpd.read_file(STATE_GEOJSON)
    possible_name_cols = [c for c in states.columns if c.lower() in {'name', 'state', 'stusps', 'state_name'}]
    name_col = 'name' if 'name' in states.columns else possible_name_cols[0]
    states['state_join'] = states[name_col].astype(str).str.strip().str.lower()
    map_df = states.merge(state_counts[['state', 'records']], left_on='state_join', right_on='state', how='left')
    map_df['records'] = map_df['records'].fillna(0)
    non_contig = {'alaska', 'hawaii', 'puerto rico'}
    map_df_plot = map_df[~map_df['state_join'].isin(non_contig)].copy()
    fig, ax = plt.subplots(figsize=(16, 10))
    map_df_plot.plot(column='records', cmap='OrRd', linewidth=0.5, edgecolor='white', legend=True, ax=ax)
    ax.set_title('Spatial concentration of reported cloud-seeding activities by state')
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'state_choropleth.png', bbox_inches='tight')
    plt.close(fig)

    # Figure 6: purpose by year stacked area for top purposes
    top_purpose_list = purpose_counts.head(5)['purpose'].tolist()
    area_df = purpose_by_year[purpose_by_year['purpose_split'].isin(top_purpose_list)].copy()
    area_pivot = area_df.pivot(index='year', columns='purpose_split', values='records').fillna(0).sort_index()
    fig, ax = plt.subplots(figsize=(14, 7))
    area_pivot.plot.area(ax=ax, colormap='tab20c')
    ax.set_title('Annual dynamics of major operational purposes')
    ax.set_xlabel('Year')
    ax.set_ylabel('Exploded records')
    ax.legend(title='Purpose', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    fig.savefig(FIG_DIR / 'purpose_by_year_area.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
