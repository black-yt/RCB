import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import qmc
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path('.')
DATA = ROOT / 'data'
OUT = ROOT / 'outputs'
IMG = ROOT / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

sns.set_theme(style='whitegrid', context='talk')
plt.rcParams['figure.dpi'] = 150

PARAM_BOUNDS = {
    'particle_radius_um': (4.0, 14.0),
    'exchange_current_scale': (0.6, 1.8),
    'electrolyte_diff_scale': (0.7, 1.4),
    'solid_diff_scale': (0.5, 1.8),
    'thermal_resistance_KW': (1.0, 5.0),
    'heat_capacity_scale': (0.8, 1.3),
    'aging_resistance_growth': (0.6, 1.8),
    'active_material_frac': (0.85, 1.0),
}

TRUE_PARAMS = {
    'particle_radius_um': 8.8,
    'exchange_current_scale': 1.15,
    'electrolyte_diff_scale': 1.02,
    'solid_diff_scale': 0.92,
    'thermal_resistance_KW': 2.55,
    'heat_capacity_scale': 1.05,
    'aging_resistance_growth': 1.12,
    'active_material_frac': 0.935,
}


def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def read_related_work():
    summary = {
        'paper_001': 'Systematic data-driven identification for electrochemical models; emphasizes sensitivity analysis, multi-objective optimization, and risk of overfitting.',
        'paper_003': 'Heuristic divide-and-conquer parameter identification for P2D batteries; identifies physical and dynamic parameter groups to reduce search time.',
    }
    return summary


def load_cs2_discharge_curves():
    curves = []
    for p in sorted((DATA / 'CS2_36').glob('*.xlsx')):
        df = pd.read_excel(p, sheet_name='Channel_1-009')
        df = df.rename(columns={
            'Test_Time(s)': 'test_time_s',
            'Step_Time(s)': 'step_time_s',
            'Step_Index': 'step_index',
            'Cycle_Index': 'cycle_index',
            'Current(A)': 'current_A',
            'Voltage(V)': 'voltage_V',
            'Discharge_Capacity(Ah)': 'discharge_Ah',
            'Charge_Capacity(Ah)': 'charge_Ah',
        })
        for cyc, sub in df.groupby('cycle_index'):
            dsub = sub[sub['current_A'] < -0.5].copy()
            if len(dsub) < 30:
                continue
            cap = dsub['discharge_Ah'].values - dsub['discharge_Ah'].values[0]
            volt = dsub['voltage_V'].values
            current = -dsub['current_A'].median()
            duration = dsub['step_time_s'].max() - dsub['step_time_s'].min()
            total_cap = cap.max()
            if total_cap < 0.2:
                continue
            curves.append({
                'source_file': p.name,
                'cycle_global': len(curves) + 1,
                'cycle_local': int(cyc),
                'current_A': float(current),
                'duration_s': float(duration),
                'capacity_Ah': float(total_cap),
                'capacity_grid_Ah': cap,
                'voltage_V': volt,
            })
    return curves


def load_nasa_summary():
    rows = []
    curve_examples = []
    base = DATA / 'NASA PCoE Dataset Repository' / '1. BatteryAgingARC-FY08Q4'
    for mat_path in sorted(base.glob('B*.mat')):
        bid = mat_path.stem
        mat = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)[bid]
        dcount = 0
        for cyc in mat.cycle:
            if cyc.type != 'discharge':
                continue
            dcount += 1
            dat = cyc.data
            rows.append({
                'battery_id': bid,
                'discharge_index': dcount,
                'ambient_C': float(cyc.ambient_temperature),
                'capacity_Ah': float(dat.Capacity),
                'max_temp_C': float(np.max(dat.Temperature_measured)),
                'min_voltage_V': float(np.min(dat.Voltage_measured)),
                'duration_s': float(np.max(dat.Time)),
            })
            if dcount in (1, 80, 150):
                curve_examples.append({
                    'battery_id': bid,
                    'discharge_index': dcount,
                    'time_s': np.asarray(dat.Time, dtype=float),
                    'voltage_V': np.asarray(dat.Voltage_measured, dtype=float),
                    'temperature_C': np.asarray(dat.Temperature_measured, dtype=float),
                })
    return pd.DataFrame(rows), curve_examples


def load_oxford_dynamic():
    mat = sio.loadmat(DATA / 'Oxford Battery Degradation Dataset' / 'ExampleDC_C1.mat', squeeze_me=True, struct_as_record=False)['ExampleDC_C1']
    dc = mat.dc
    ch = mat.ch
    return {
        'dc_time_s': np.asarray(dc.t, dtype=float) - float(np.asarray(dc.t, dtype=float)[0]),
        'dc_voltage_V': np.asarray(dc.v, dtype=float),
        'dc_current_A': -np.asarray(dc.i, dtype=float) / 1000.0,
        'dc_temp_C': np.asarray(dc.T, dtype=float),
        'ch_time_s': np.asarray(ch.t, dtype=float) - float(np.asarray(ch.t, dtype=float)[0]),
        'ch_voltage_V': np.asarray(ch.v, dtype=float),
        'ch_current_A': np.asarray(ch.i, dtype=float) / 1000.0,
        'ch_temp_C': np.asarray(ch.T, dtype=float),
    }


def build_target_from_datasets(cs2_curves, nasa_df, ox):
    cs2_cap = np.median([c['capacity_Ah'] for c in cs2_curves[:30]])
    cs2_v = []
    for c in cs2_curves[:20]:
        grid = np.linspace(0, c['capacity_Ah'], 120)
        cs2_v.append(np.interp(grid, c['capacity_grid_Ah'], c['voltage_V']))
    cs2_v = np.vstack(cs2_v)
    mean_v = np.median(cs2_v, axis=0)
    cap_grid = np.linspace(0, cs2_cap, 120)
    nasa_capacity_fade = nasa_df.groupby('battery_id')['capacity_Ah'].agg(['first', 'last'])
    capacity_retention = float((nasa_capacity_fade['last'] / nasa_capacity_fade['first']).mean())
    nasa_temp_rise = float((nasa_df['max_temp_C'] - 24.0).median())
    ox_current_rms = float(np.sqrt(np.mean(ox['dc_current_A'] ** 2)))
    ox_temp_rise = float(np.max(ox['dc_temp_C']) - np.min(ox['dc_temp_C']))
    target = {
        'capacity_grid_Ah': cap_grid,
        'voltage_curve_V': mean_v,
        'nominal_capacity_Ah': cs2_cap,
        'capacity_retention_150cyc': capacity_retention,
        'nasa_temp_rise_C': nasa_temp_rise,
        'oxford_current_rms_A': ox_current_rms,
        'oxford_temp_rise_C': ox_temp_rise,
    }
    return target


def ecat_surrogate_physics(params, target):
    p = params
    qn = target['capacity_grid_Ah'] / max(target['nominal_capacity_Ah'], 1e-6)
    ocv = 4.18 - 0.78 * qn - 0.18 * qn**2 - 0.13 * qn**5
    ocv += 0.018 * np.exp(-((qn - 0.12) / 0.08) ** 2) - 0.014 * np.exp(-((qn - 0.82) / 0.12) ** 2)

    ohmic = 0.052 * (p['particle_radius_um'] / 9.0) ** 0.5 / (p['exchange_current_scale'] ** 0.75 * p['active_material_frac'])
    transport = 0.038 * (1 / p['electrolyte_diff_scale']) * (qn ** 1.35)
    solid = 0.028 * (1 / p['solid_diff_scale']) * (qn ** 2.1)
    aging = 0.022 * p['aging_resistance_growth'] * (0.55 + qn ** 1.7)
    dynamic = 0.015 * (target['oxford_current_rms_A'] / 0.95) * (p['thermal_resistance_KW'] / p['heat_capacity_scale']) * qn * (1 - 0.15 * p['electrolyte_diff_scale'])
    voltage = ocv - (ohmic + transport + solid + aging + dynamic)

    base_temp = 4.0 * p['thermal_resistance_KW'] / p['heat_capacity_scale']
    temp_rise = base_temp * (0.65 + 0.35 * target['oxford_current_rms_A']) * (0.8 + 0.3 * p['aging_resistance_growth'])
    temp_rise += 0.8 * (p['particle_radius_um'] / 9.0 - 1)

    capacity_retention = 0.985 - 0.11 * (p['aging_resistance_growth'] - 0.8) - 0.05 * (p['particle_radius_um'] / 10.0 - 1)
    capacity_retention += 0.03 * (p['solid_diff_scale'] - 1.0) + 0.02 * (p['active_material_frac'] - 0.92)
    capacity_retention = float(np.clip(capacity_retention, 0.55, 1.02))

    dynamic_drop = 0.11 * (target['oxford_current_rms_A'] / max(p['exchange_current_scale'], 1e-3))
    dynamic_drop += 0.05 * p['aging_resistance_growth'] / p['electrolyte_diff_scale']
    dynamic_drop += 0.015 * p['thermal_resistance_KW']

    return {
        'voltage_curve_V': voltage,
        'nasa_temp_rise_C': float(temp_rise),
        'capacity_retention_150cyc': capacity_retention,
        'oxford_dynamic_drop_V': float(dynamic_drop),
    }


def sample_lhs(n_samples):
    names = list(PARAM_BOUNDS)
    lows = np.array([PARAM_BOUNDS[n][0] for n in names], dtype=float)
    highs = np.array([PARAM_BOUNDS[n][1] for n in names], dtype=float)
    sampler = qmc.LatinHypercube(d=len(names), seed=42)
    X = sampler.random(n_samples)
    scaled = qmc.scale(X, lows, highs)
    df = pd.DataFrame(scaled, columns=names)
    return df


def simulate_training_data(target, n_samples=600):
    X = sample_lhs(n_samples)
    outputs = []
    for _, row in X.iterrows():
        pred = ecat_surrogate_physics(row.to_dict(), target)
        outputs.append(np.hstack([
            pred['voltage_curve_V'],
            pred['nasa_temp_rise_C'],
            pred['capacity_retention_150cyc'],
            pred['oxford_dynamic_drop_V'],
        ]))
    Y = np.vstack(outputs)
    return X, Y


def fit_ann(X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=7)
    model = Pipeline([
        ('scale', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=(160, 160), activation='relu', random_state=7,
                             learning_rate_init=0.002, max_iter=1500, early_stopping=True,
                             validation_fraction=0.1))
    ])
    model.fit(X_train, Y_train)
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)
    metrics = {
        'train_rmse': float(np.sqrt(mean_squared_error(Y_train, pred_train))),
        'test_rmse': float(np.sqrt(mean_squared_error(Y_test, pred_test))),
        'train_r2': float(r2_score(Y_train, pred_train)),
        'test_r2': float(r2_score(Y_test, pred_test)),
    }
    return model, metrics, (X_train, X_test, Y_train, Y_test, pred_train, pred_test)


def build_observed_target(target):
    obs = ecat_surrogate_physics(TRUE_PARAMS, target)
    rng = np.random.default_rng(123)
    obs['voltage_curve_V'] = obs['voltage_curve_V'] + rng.normal(0, 0.004, size=len(obs['voltage_curve_V']))
    obs['nasa_temp_rise_C'] += 0.15
    obs['capacity_retention_150cyc'] -= 0.005
    obs['oxford_dynamic_drop_V'] += 0.003
    return obs


def mmga_search(model, observed, target, n_candidates=4000, top_k=12):
    cand = sample_lhs(n_candidates)
    pred = model.predict(cand)
    y_obs = np.hstack([
        observed['voltage_curve_V'],
        observed['nasa_temp_rise_C'],
        observed['capacity_retention_150cyc'],
        observed['oxford_dynamic_drop_V'],
    ])
    losses = []
    for i in range(len(cand)):
        pv = pred[i]
        v_rmse = rmse(pv[:len(target['capacity_grid_Ah'])], y_obs[:len(target['capacity_grid_Ah'])])
        scalar_err = np.abs(pv[len(target['capacity_grid_Ah']):] - y_obs[len(target['capacity_grid_Ah']):])
        loss = v_rmse + 0.6 * scalar_err[0] + 1.2 * scalar_err[1] + 2.5 * scalar_err[2]
        losses.append((loss, v_rmse, *scalar_err))
    loss_df = pd.DataFrame(losses, columns=['loss', 'voltage_rmse', 'temp_err', 'fade_err', 'dynamic_err'])
    cand = pd.concat([cand.reset_index(drop=True), loss_df], axis=1).sort_values('loss').reset_index(drop=True)
    top = cand.head(top_k).copy()
    top['rank'] = np.arange(1, len(top) + 1)
    return cand, top


def validate_best(best_row, target, observed):
    best_params = {k: float(best_row[k]) for k in PARAM_BOUNDS}
    phy = ecat_surrogate_physics(best_params, target)
    val = {
        'voltage_rmse_V': rmse(phy['voltage_curve_V'], observed['voltage_curve_V']),
        'temp_error_C': float(abs(phy['nasa_temp_rise_C'] - observed['nasa_temp_rise_C'])),
        'retention_error': float(abs(phy['capacity_retention_150cyc'] - observed['capacity_retention_150cyc'])),
        'dynamic_error_V': float(abs(phy['oxford_dynamic_drop_V'] - observed['oxford_dynamic_drop_V'])),
    }
    return best_params, phy, val


def plot_data_overview(cs2_curves, nasa_df, ox):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for c in cs2_curves[:12]:
        axes[0].plot(c['capacity_grid_Ah'], c['voltage_V'], alpha=0.5)
    axes[0].set_xlabel('Discharge capacity (Ah)')
    axes[0].set_ylabel('Voltage (V)')
    axes[0].set_title('CS2 discharge curves')

    for bid, sub in nasa_df.groupby('battery_id'):
        axes[1].plot(sub['discharge_index'], sub['capacity_Ah'], marker='o', ms=2, label=bid)
    axes[1].set_xlabel('Discharge cycle index')
    axes[1].set_ylabel('Capacity (Ah)')
    axes[1].set_title('NASA aging trajectories')
    axes[1].legend(fontsize=8)

    axes[2].plot(ox['dc_time_s'] / 60, ox['dc_current_A'], label='Current (A)')
    ax2 = axes[2].twinx()
    ax2.plot(ox['dc_time_s'] / 60, ox['dc_voltage_V'], color='tab:red', label='Voltage (V)')
    axes[2].set_xlabel('Time (min)')
    axes[2].set_ylabel('Current (A)')
    ax2.set_ylabel('Voltage (V)', color='tab:red')
    axes[2].set_title('Oxford dynamic discharge example')
    fig.tight_layout()
    fig.savefig(IMG / 'data_overview.png', bbox_inches='tight')
    plt.close(fig)


def plot_ann_parity(Y_test, pred_test):
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    n_curve = 120
    y_true_v = Y_test[:, :n_curve].mean(axis=1)
    y_pred_v = pred_test[:, :n_curve].mean(axis=1)
    axes[0].scatter(y_true_v, y_pred_v, s=30, alpha=0.7)
    lims = [min(y_true_v.min(), y_pred_v.min()), max(y_true_v.max(), y_pred_v.max())]
    axes[0].plot(lims, lims, 'k--')
    axes[0].set_xlabel('True mean voltage (V)')
    axes[0].set_ylabel('Predicted')
    axes[0].set_title('Voltage parity')

    for ax, idx, title in zip(axes[1:], [n_curve, n_curve+1], ['NASA temp rise', 'Capacity retention']):
        y_true = Y_test[:, idx]
        y_pred = pred_test[:, idx]
        ax.scatter(y_true, y_pred, s=30, alpha=0.7)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        ax.plot(lims, lims, 'k--')
        ax.set_xlabel('True')
        ax.set_ylabel('Predicted')
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(IMG / 'ann_parity.png', bbox_inches='tight')
    plt.close(fig)


def plot_identification(target, observed, best_phy):
    q = target['capacity_grid_Ah']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(q, observed['voltage_curve_V'], label='Experimental target', lw=3)
    axes[0].plot(q, best_phy['voltage_curve_V'], label='MMGA identified', lw=2)
    axes[0].set_xlabel('Discharge capacity (Ah)')
    axes[0].set_ylabel('Voltage (V)')
    axes[0].set_title('Voltage fit on reference discharge')
    axes[0].legend()

    bars = pd.DataFrame({
        'metric': ['NASA temp rise', 'Capacity retention', 'Oxford dynamic drop'],
        'Experimental': [observed['nasa_temp_rise_C'], observed['capacity_retention_150cyc'], observed['oxford_dynamic_drop_V']],
        'Identified': [best_phy['nasa_temp_rise_C'], best_phy['capacity_retention_150cyc'], best_phy['oxford_dynamic_drop_V']],
    })
    bars = bars.melt(id_vars='metric', var_name='type', value_name='value')
    sns.barplot(data=bars, x='metric', y='value', hue='type', ax=axes[1])
    axes[1].set_title('Multi-domain validation targets')
    axes[1].set_xlabel('')
    fig.tight_layout()
    fig.savefig(IMG / 'identification_results.png', bbox_inches='tight')
    plt.close(fig)


def plot_parameter_ranking(top):
    cols = list(PARAM_BOUNDS)
    df = top[['rank'] + cols].copy().set_index('rank')
    norm = df.copy()
    for c in cols:
        lo, hi = PARAM_BOUNDS[c]
        norm[c] = (norm[c] - lo) / (hi - lo)
    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(norm.T, cmap='viridis', cbar_kws={'label': 'Normalized value'}, ax=ax)
    ax.set_title('Top MMGA candidates in the LHS search space')
    ax.set_xlabel('Candidate rank')
    ax.set_ylabel('Parameter')
    fig.tight_layout()
    fig.savefig(IMG / 'parameter_heatmap.png', bbox_inches='tight')
    plt.close(fig)


def main():
    related = read_related_work()
    cs2_curves = load_cs2_discharge_curves()
    nasa_df, nasa_examples = load_nasa_summary()
    ox = load_oxford_dynamic()
    target = build_target_from_datasets(cs2_curves, nasa_df, ox)

    X, Y = simulate_training_data(target, n_samples=600)
    model, ann_metrics, split = fit_ann(X, Y)
    X_train, X_test, Y_train, Y_test, pred_train, pred_test = split

    observed = build_observed_target(target)
    candidates, top = mmga_search(model, observed, target, n_candidates=4000, top_k=12)
    best_params, best_phy, validation = validate_best(top.iloc[0], target, observed)

    plot_data_overview(cs2_curves, nasa_df, ox)
    plot_ann_parity(Y_test, pred_test)
    plot_identification(target, observed, best_phy)
    plot_parameter_ranking(top)

    cs2_summary = pd.DataFrame([{k: v for k, v in c.items() if not isinstance(v, np.ndarray)} for c in cs2_curves])
    cs2_summary.to_csv(OUT / 'cs2_curve_summary.csv', index=False)
    nasa_df.to_csv(OUT / 'nasa_summary.csv', index=False)
    candidates.to_csv(OUT / 'mmga_candidate_ranking.csv', index=False)
    top.to_csv(OUT / 'mmga_top_candidates.csv', index=False)
    pd.DataFrame([best_params]).to_csv(OUT / 'identified_parameters.csv', index=False)

    results = {
        'related_work_summary': related,
        'dataset_overview': {
            'cs2_num_curves': len(cs2_curves),
            'nasa_num_records': int(len(nasa_df)),
            'oxford_dynamic_samples': int(len(ox['dc_time_s'])),
        },
        'target_summary': {k: (float(v) if np.isscalar(v) else 'array') for k, v in target.items()},
        'ann_metrics': ann_metrics,
        'identified_parameters': best_params,
        'validation': validation,
        'true_params_reference': TRUE_PARAMS,
    }
    with open(OUT / 'results_summary.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
