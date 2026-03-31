import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.stats import qmc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)
sns.set_theme(style="whitegrid")

ROOT = Path('.')
DATA = ROOT / 'data'
OUTPUTS = ROOT / 'outputs'
REPORT_IMG = ROOT / 'report' / 'images'
OUTPUTS.mkdir(exist_ok=True, parents=True)
REPORT_IMG.mkdir(exist_ok=True, parents=True)

GRID = np.linspace(0.0, 1.0, 120)

PARAM_BOUNDS = {
    'r_p_um': (3.0, 12.0),
    'k_ref': (0.6, 1.8),
    'd_sei': (0.0, 0.25),
    'r_ohm': (0.01, 0.08),
    'tau_diff': (0.3, 2.2),
    'thermal_gain': (0.5, 5.0),
    'thermal_loss': (0.1, 1.2),
    'q_loss': (0.0, 0.22),
}
PARAM_NAMES = list(PARAM_BOUNDS.keys())


def as_list(x):
    if isinstance(x, np.ndarray):
        return np.asarray(x).squeeze()
    return np.asarray(x)


def resample_curve(x, y, n=120):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 5:
        return np.full(n, np.nan)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x = x - x.min()
    if x.max() <= 0:
        return np.full(n, y.mean())
    xn = x / x.max()
    uniq, idx = np.unique(xn, return_index=True)
    yy = y[idx]
    f = interp1d(uniq, yy, kind='linear', fill_value='extrapolate')
    return f(np.linspace(0, 1, n))


def normalize_capacity(q):
    q = np.asarray(q, dtype=float)
    q = q - np.nanmin(q)
    denom = np.nanmax(q) - np.nanmin(q)
    if denom <= 1e-12:
        return np.zeros_like(q)
    return q / denom


def extract_cs2_reference():
    frames = []
    for fn in sorted((DATA / 'CS2_36').glob('*.xlsx')):
        df = pd.read_excel(fn, sheet_name='Channel_1-009')
        discharge = df[df['Current(A)'] < -0.5].copy()
        if discharge.empty:
            continue
        for cyc, g in discharge.groupby('Cycle_Index'):
            g = g.sort_values('Test_Time(s)')
            cap_raw = g['Discharge_Capacity(Ah)'].to_numpy(dtype=float)
            cap = cap_raw - np.nanmin(cap_raw)
            if np.nanmax(cap) < 0.1:
                continue
            qn = normalize_capacity(cap)
            frames.append({
                'source_file': fn.name,
                'cycle_index': int(cyc),
                'time_s': g['Test_Time(s)'].to_numpy(dtype=float) - g['Test_Time(s)'].iloc[0],
                'voltage_v': g['Voltage(V)'].to_numpy(dtype=float),
                'current_a': g['Current(A)'].to_numpy(dtype=float),
                'capacity_ah': cap,
                'capacity_norm': qn,
                'temperature_c': np.full(len(g), np.nan),
            })
    if not frames:
        raise RuntimeError('No CS2 discharge cycles found.')
    target = max(frames, key=lambda d: np.nanmax(d['capacity_ah']))
    target['dataset'] = 'CS2'
    return target, frames


def extract_nasa_reference():
    path = DATA / 'NASA PCoE Dataset Repository' / '1. BatteryAgingARC-FY08Q4' / 'B0005.mat'
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)['B0005']
    discharge_cycles = []
    for cyc in np.atleast_1d(mat.cycle):
        if getattr(cyc, 'type', None) != 'discharge':
            continue
        data = cyc.data
        cap = getattr(data, 'Capacity', np.nan)
        time_s = as_list(getattr(data, 'Time'))
        volt = as_list(getattr(data, 'Voltage_measured'))
        temp = as_list(getattr(data, 'Temperature_measured'))
        if len(time_s) < 20:
            continue
        cap_arr = np.asarray(cap).squeeze()
        cap_val = float(cap_arr) if np.size(cap_arr) == 1 else float(np.nanmax(cap_arr))
        discharge_cycles.append({
            'capacity': cap_val,
            'time_s': time_s,
            'voltage_v': volt,
            'temperature_c': temp,
            'current_a': as_list(getattr(data, 'Current_measured')),
        })
    target = max(discharge_cycles, key=lambda d: d['capacity'])
    q_proxy = np.linspace(0, target['capacity'], len(target['time_s']))
    target.update({
        'dataset': 'NASA',
        'capacity_ah': q_proxy,
        'capacity_norm': normalize_capacity(q_proxy),
    })
    return target, discharge_cycles


def extract_oxford_reference():
    mat = loadmat(DATA / 'Oxford Battery Degradation Dataset' / 'ExampleDC_C1.mat', squeeze_me=True, struct_as_record=False)['ExampleDC_C1']
    dc = mat.dc
    q = np.abs(as_list(dc.q)) / 1000.0
    ref = {
        'dataset': 'Oxford',
        'time_s': as_list(dc.t) - np.min(as_list(dc.t)),
        'voltage_v': as_list(dc.v),
        'temperature_c': as_list(dc.T),
        'current_a': as_list(dc.i) / 1000.0,
        'capacity_ah': q,
        'capacity_norm': normalize_capacity(q),
    }
    return ref


def summarize_reference(ref):
    return {
        'dataset': ref['dataset'],
        'n_points': int(len(ref['time_s'])),
        'duration_s': float(np.nanmax(ref['time_s']) - np.nanmin(ref['time_s'])),
        'voltage_min_v': float(np.nanmin(ref['voltage_v'])),
        'voltage_max_v': float(np.nanmax(ref['voltage_v'])),
        'capacity_max_ah': float(np.nanmax(ref['capacity_ah'])),
        'temperature_min_c': float(np.nanmin(ref['temperature_c'])) if np.isfinite(ref['temperature_c']).any() else None,
        'temperature_max_c': float(np.nanmax(ref['temperature_c'])) if np.isfinite(ref['temperature_c']).any() else None,
    }


def simulate_proxy(params, load_profile):
    p = params
    x = GRID
    current_scale = max(0.5, float(np.nanmean(np.abs(load_profile['current_a']))))
    ambient = 25.0 if not np.isfinite(load_profile['temperature_c']).any() else float(np.nanmin(load_profile['temperature_c']))
    capacity_nom = load_profile['capacity_max_ah']
    q_eff = capacity_nom * (1.0 - p['q_loss'])
    q_eff = max(q_eff, 0.35 * capacity_nom)

    ocv = 4.2 - 0.95 * x - 0.18 * np.tanh((x - 0.82) / 0.06)
    kinetics = 0.07 * current_scale / p['k_ref'] * np.sqrt(x + 0.03)
    diffusion = 0.045 * p['tau_diff'] * (1 - np.exp(-4.5 * x))
    sei = 0.06 * p['d_sei'] * x ** 1.4
    ohmic = current_scale * p['r_ohm']
    particle = 0.012 * (p['r_p_um'] / 10.0) * x
    voltage = ocv - kinetics - diffusion - sei - ohmic - particle
    voltage = np.clip(voltage, 2.5, 4.25)

    t_norm = x * load_profile['duration_s'] * (capacity_nom / q_eff)
    heat_source = p['thermal_gain'] * (current_scale ** 2) * (0.35 + 0.65 * x)
    cooling = p['thermal_loss'] * np.sqrt(t_norm / max(load_profile['duration_s'], 1.0))
    temp = ambient + heat_source - cooling

    capacity = x * q_eff
    return {
        'grid': x,
        'time_s': t_norm,
        'voltage_v': voltage,
        'temperature_c': temp,
        'capacity_ah': capacity,
        'capacity_norm': normalize_capacity(capacity),
    }


def featureize_curve(curve):
    v = curve['voltage_v']
    t = curve['temperature_c']
    q = curve['capacity_ah']
    feats = {
        'capacity_max': float(np.nanmax(q)),
        'v_start': float(v[0]),
        'v_mid': float(v[len(v)//2]),
        'v_end': float(v[-1]),
        'v_mean': float(np.nanmean(v)),
        'v_slope': float(v[0] - v[-1]),
        'temp_rise': float(np.nan_to_num(t[-1] - t[0], nan=0.0)),
        'temp_max': float(np.nan_to_num(np.nanmax(t), nan=25.0)),
    }
    for idx, frac in enumerate([0.1,0.25,0.5,0.75,0.9]):
        feats[f'v_q{idx+1}'] = float(np.interp(frac, curve['capacity_norm'], v))
        feats[f't_q{idx+1}'] = float(np.interp(frac, curve['capacity_norm'], t))
    return feats


def build_lhs_library(load_profile, n_samples=2500):
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=SEED)
    sample = sampler.random(n=n_samples)
    lower = np.array([PARAM_BOUNDS[k][0] for k in PARAM_NAMES])
    upper = np.array([PARAM_BOUNDS[k][1] for k in PARAM_NAMES])
    X = qmc.scale(sample, lower, upper)
    rows = []
    curves = {'voltage': [], 'temperature': [], 'capacity': [], 'time': []}
    for x in X:
        params = {k: float(v) for k, v in zip(PARAM_NAMES, x)}
        curve = simulate_proxy(params, load_profile)
        rows.append({**params, **featureize_curve(curve)})
        curves['voltage'].append(curve['voltage_v'])
        curves['temperature'].append(curve['temperature_c'])
        curves['capacity'].append(curve['capacity_ah'])
        curves['time'].append(curve['time_s'])
    return pd.DataFrame(rows), {k: np.asarray(v) for k, v in curves.items()}


def train_surrogate(df):
    X = df[PARAM_NAMES].to_numpy()
    feature_cols = [c for c in df.columns if c not in PARAM_NAMES]
    y = df[feature_cols].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPRegressor(hidden_layer_sizes=(64, 64), activation='relu', random_state=SEED,
                             max_iter=350, early_stopping=True, validation_fraction=0.15, tol=1e-4))
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    metrics = {
        'r2_mean': float(r2_score(y_test, pred, multioutput='uniform_average')),
        'mae_mean': float(mean_absolute_error(y_test, pred)),
        'rmse_mean': float(np.sqrt(mean_squared_error(y_test, pred))),
    }
    per_feature = {}
    for i, c in enumerate(feature_cols):
        per_feature[c] = {
            'r2': float(r2_score(y_test[:, i], pred[:, i])),
            'mae': float(mean_absolute_error(y_test[:, i], pred[:, i])),
        }
    return model, feature_cols, metrics, per_feature, (y_test, pred)


def fit_score(observed_curve, candidate_curve):
    v_obs = resample_curve(observed_curve['capacity_norm'], observed_curve['voltage_v'], len(GRID))
    v_sim = resample_curve(candidate_curve['capacity_norm'], candidate_curve['voltage_v'], len(GRID))
    t_obs = resample_curve(observed_curve['capacity_norm'], np.nan_to_num(observed_curve['temperature_c'], nan=np.nanmedian(candidate_curve['temperature_c'])), len(GRID))
    t_sim = resample_curve(candidate_curve['capacity_norm'], candidate_curve['temperature_c'], len(GRID))
    cap_err = abs(np.nanmax(observed_curve['capacity_ah']) - np.nanmax(candidate_curve['capacity_ah']))
    v_rmse = float(np.sqrt(np.nanmean((v_obs - v_sim) ** 2)))
    t_rmse = float(np.sqrt(np.nanmean((t_obs - t_sim) ** 2)))
    return 0.7 * v_rmse + 0.2 * t_rmse + 0.1 * cap_err, {'v_rmse': v_rmse, 't_rmse': t_rmse, 'cap_abs_err': float(cap_err)}


def feature_objective_from_params(model, params_vec, target_features):
    pred = model.predict(np.asarray(params_vec, dtype=float).reshape(1, -1))[0]
    return float(np.sqrt(np.mean((pred - target_features) ** 2)))


def search_parameters(model, feature_names, target_curve, load_profile, n_iter=450, elite_frac=0.15):
    target_features = np.array([featureize_curve(target_curve)[c] for c in feature_names], dtype=float)
    sampler = qmc.LatinHypercube(d=len(PARAM_NAMES), seed=SEED + 1)
    sample = sampler.random(n=n_iter)
    lower = np.array([PARAM_BOUNDS[k][0] for k in PARAM_NAMES])
    upper = np.array([PARAM_BOUNDS[k][1] for k in PARAM_NAMES])
    X = qmc.scale(sample, lower, upper)
    approx_scores = []
    for row in X:
        approx_scores.append(feature_objective_from_params(model, row, target_features))
    order = np.argsort(approx_scores)
    top = X[order[:max(40, int(elite_frac * len(X)))]]
    elite_scores = []
    for row in top:
        params = {k: float(v) for k, v in zip(PARAM_NAMES, row)}
        curve = simulate_proxy(params, load_profile)
        score, metrics = fit_score(target_curve, curve)
        elite_scores.append((score, params, metrics, curve))
    elite_scores.sort(key=lambda x: x[0])
    elite_arr = np.array([[e[1][k] for k in PARAM_NAMES] for e in elite_scores[:max(10, len(elite_scores)//3)]])
    mu = elite_arr.mean(axis=0)
    sigma = elite_arr.std(axis=0) + 1e-6
    refined = []
    rng = np.random.default_rng(SEED + 2)
    for _ in range(250):
        cand = np.clip(rng.normal(mu, sigma), lower, upper)
        params = {k: float(v) for k, v in zip(PARAM_NAMES, cand)}
        curve = simulate_proxy(params, load_profile)
        score, metrics = fit_score(target_curve, curve)
        refined.append((score, params, metrics, curve))
    best = min(elite_scores + refined, key=lambda x: x[0])
    return best


def make_data_overview_plot(cs2, nasa, oxford):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    refs = [cs2, nasa, oxford]
    for ax, ref in zip(axes, refs):
        ax.plot(ref['time_s'] / 60.0, ref['voltage_v'], label='Voltage', color='tab:blue')
        ax.set_title(ref['dataset'])
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Voltage (V)', color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax2 = ax.twinx()
        if np.isfinite(ref['temperature_c']).any():
            ax2.plot(ref['time_s'] / 60.0, ref['temperature_c'], label='Temperature', color='tab:red', alpha=0.7)
            ax2.set_ylabel('Temperature (°C)', color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
    fig.suptitle('Experimental discharge profiles used in the proxy MMGA study')
    fig.savefig(REPORT_IMG / 'data_overview.png', dpi=200)
    plt.close(fig)


def make_surrogate_parity_plot(y_test, pred, feature_names):
    sel = ['capacity_max', 'v_end', 'temp_max']
    idx = [feature_names.index(s) for s in sel]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, i, name in zip(axes, idx, sel):
        ax.scatter(y_test[:, i], pred[:, i], s=10, alpha=0.5)
        low = min(y_test[:, i].min(), pred[:, i].min())
        high = max(y_test[:, i].max(), pred[:, i].max())
        ax.plot([low, high], [low, high], 'k--', lw=1)
        ax.set_xlabel('Ground truth')
        ax.set_ylabel('Prediction')
        ax.set_title(name)
    fig.suptitle('ANN surrogate parity on held-out synthetic library')
    fig.savefig(REPORT_IMG / 'surrogate_parity.png', dpi=200)
    plt.close(fig)


def make_fit_plot(target, best_curve):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    x_obs = resample_curve(target['capacity_norm'], target['capacity_ah'], len(GRID))
    axes[0].plot(x_obs, resample_curve(target['capacity_norm'], target['voltage_v'], len(GRID)), label='Observed', lw=2)
    axes[0].plot(best_curve['capacity_ah'], best_curve['voltage_v'], label='Proxy fit', lw=2)
    axes[0].set_xlabel('Capacity (Ah)')
    axes[0].set_ylabel('Voltage (V)')
    axes[0].legend()
    axes[0].set_title('CS2 voltage-capacity fit')
    axes[1].plot(x_obs, resample_curve(target['capacity_norm'], np.nan_to_num(target['temperature_c'], nan=np.nanmedian(best_curve['temperature_c'])), len(GRID)), label='Observed', lw=2)
    axes[1].plot(best_curve['capacity_ah'], best_curve['temperature_c'], label='Proxy fit', lw=2)
    axes[1].set_xlabel('Capacity (Ah)')
    axes[1].set_ylabel('Temperature (°C)')
    axes[1].legend()
    axes[1].set_title('CS2 temperature-capacity fit')
    fig.savefig(REPORT_IMG / 'cs2_fit.png', dpi=200)
    plt.close(fig)


def make_external_validation_plot(nasa, nasa_curve, oxford, oxford_curve):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    pairs = [(nasa, nasa_curve, 'NASA'), (oxford, oxford_curve, 'Oxford')]
    for row, (obs, sim, title) in enumerate(pairs):
        x_obs = resample_curve(obs['capacity_norm'], obs['capacity_ah'], len(GRID))
        axes[row, 0].plot(x_obs, resample_curve(obs['capacity_norm'], obs['voltage_v'], len(GRID)), label='Observed', lw=2)
        axes[row, 0].plot(sim['capacity_ah'], sim['voltage_v'], label='Proxy fit', lw=2)
        axes[row, 0].set_title(f'{title} voltage-capacity')
        axes[row, 0].set_xlabel('Capacity (Ah)')
        axes[row, 0].set_ylabel('Voltage (V)')
        axes[row, 0].legend()
        axes[row, 1].plot(x_obs, resample_curve(obs['capacity_norm'], np.nan_to_num(obs['temperature_c'], nan=np.nanmedian(sim['temperature_c'])), len(GRID)), label='Observed', lw=2)
        axes[row, 1].plot(sim['capacity_ah'], sim['temperature_c'], label='Proxy fit', lw=2)
        axes[row, 1].set_title(f'{title} temperature-capacity')
        axes[row, 1].set_xlabel('Capacity (Ah)')
        axes[row, 1].set_ylabel('Temperature (°C)')
        axes[row, 1].legend()
    fig.savefig(REPORT_IMG / 'external_validation.png', dpi=200)
    plt.close(fig)


def main():
    cs2_target, cs2_cycles = extract_cs2_reference()
    nasa_target, nasa_cycles = extract_nasa_reference()
    oxford_target = extract_oxford_reference()

    summaries = {
        'CS2': summarize_reference(cs2_target),
        'NASA': summarize_reference(nasa_target),
        'Oxford': summarize_reference(oxford_target),
        'n_cs2_cycles': len(cs2_cycles),
        'n_nasa_discharge_cycles': len(nasa_cycles),
    }
    with open(OUTPUTS / 'data_summary.json', 'w') as f:
        json.dump(summaries, f, indent=2)

    make_data_overview_plot(cs2_target, nasa_target, oxford_target)

    load_profile = {
        'duration_s': summaries['CS2']['duration_s'],
        'capacity_max_ah': summaries['CS2']['capacity_max_ah'],
        'current_a': cs2_target['current_a'],
        'temperature_c': cs2_target['temperature_c'],
    }

    df_lib, lib_curves = build_lhs_library(load_profile, n_samples=900)
    df_lib.to_csv(OUTPUTS / 'lhs_samples.csv', index=False)
    np.savez(OUTPUTS / 'synthetic_library.npz', **lib_curves)

    model, feature_names, metrics, per_feature, holdout = train_surrogate(df_lib)
    with open(OUTPUTS / 'surrogate_metrics.json', 'w') as f:
        json.dump({'aggregate': metrics, 'per_feature': per_feature}, f, indent=2)
    make_surrogate_parity_plot(holdout[0], holdout[1], feature_names)

    best_cs2 = search_parameters(model, feature_names, cs2_target, load_profile)
    best_nasa = search_parameters(model, feature_names, nasa_target, {
        'duration_s': summaries['NASA']['duration_s'],
        'capacity_max_ah': summaries['NASA']['capacity_max_ah'],
        'current_a': nasa_target['current_a'],
        'temperature_c': nasa_target['temperature_c'],
    })
    best_oxford = search_parameters(model, feature_names, oxford_target, {
        'duration_s': summaries['Oxford']['duration_s'],
        'capacity_max_ah': summaries['Oxford']['capacity_max_ah'],
        'current_a': oxford_target['current_a'],
        'temperature_c': oxford_target['temperature_c'],
    })

    rows = []
    for dataset, result in [('CS2', best_cs2), ('NASA', best_nasa), ('Oxford', best_oxford)]:
        score, params, fit_metrics, _ = result
        rows.append({'dataset': dataset, 'objective': score, **fit_metrics, **params})
    pd.DataFrame(rows).to_csv(OUTPUTS / 'identified_parameters.csv', index=False)

    make_fit_plot(cs2_target, best_cs2[3])
    make_external_validation_plot(nasa_target, best_nasa[3], oxford_target, best_oxford[3])

    with open(OUTPUTS / 'fit_metrics.json', 'w') as f:
        json.dump({
            'CS2': {'objective': best_cs2[0], **best_cs2[2]},
            'NASA': {'objective': best_nasa[0], **best_nasa[2]},
            'Oxford': {'objective': best_oxford[0], **best_oxford[2]},
        }, f, indent=2)

    print('Completed proxy MMGA experiment.')
    print(json.dumps({'surrogate': metrics, 'fits': {
        'CS2': best_cs2[2], 'NASA': best_nasa[2], 'Oxford': best_oxford[2]
    }}, indent=2))


if __name__ == '__main__':
    main()
