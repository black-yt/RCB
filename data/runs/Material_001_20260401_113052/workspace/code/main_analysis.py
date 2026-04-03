import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / 'data' / 'M-AI-Synth__Materials_AI_Dataset_.txt'
OUT_DIR = BASE_DIR / 'outputs'
FIG_DIR = BASE_DIR / 'report' / 'images'

OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():
    """Parse the synthetic multi-workflow dataset from the text file.

    The file contains three logical blocks corresponding to
    property prediction, structure generation, and autonomous
    experimental optimization workflows.
    """
    text = DATA_PATH.read_text(encoding='utf-8').strip().splitlines()

    # Helper to parse the JSON-like lists present in the file
    def parse_list(line):
        return np.array(json.loads(line.strip().rstrip(',')))

    # We know the layout from inspecting the file structure
    # Lines are 1-indexed in the description; here they are 0-indexed
    # Block 1: property_prediction
    comp_feats = parse_list(text[1])        # composition/descriptor length N
    process_feats = parse_list(text[2])     # processing/environmental feature length N
    graph_edges = parse_list(text[3])       # simplified graph connectivity
    target_props = parse_list(text[4])      # target material property

    # Block 2: structure_generation
    struct_latent_1 = parse_list(text[7])   # latent representation set A
    struct_latent_2 = parse_list(text[8])   # latent representation set B

    # Block 3: autonomous_optimization
    temp_range = parse_list(text[11])       # exploration temperature range
    time_range = parse_list(text[12])       # exploration time range
    best_temp = parse_list(text[13])[0]
    best_time = parse_list(text[14])[0]
    best_yield = parse_list(text[15])[0]
    target_yield = parse_list(text[16])[0]

    return {
        'property_prediction': {
            'comp_feats': comp_feats,
            'process_feats': process_feats,
            'graph_edges': graph_edges,
            'target_props': target_props,
        },
        'structure_generation': {
            'latent_a': struct_latent_1,
            'latent_b': struct_latent_2,
        },
        'optimization': {
            'temp_range': temp_range,
            'time_range': time_range,
            'best_temp': best_temp,
            'best_time': best_time,
            'best_yield': best_yield,
            'target_yield': target_yield,
        },
    }


def run_property_prediction(block):
    """Simple supervised regression on synthetic 1D features.

    We build a small feature matrix using composition and process
    features and train/test split with a linear model to predict the
    target property.
    """
    x1 = block['comp_feats']
    x2 = block['process_feats']
    y = block['target_props']

    # ensure consistent shape
    n = min(len(x1), len(x2), len(y))
    x1, x2, y = x1[:n], x2[:n], y[:n]

    X = np.stack([x1, x2, x1 * x2], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        'train_rmse': float(np.sqrt(mean_squared_error(y_train, y_train_pred))),
        'test_rmse': float(np.sqrt(mean_squared_error(y_test, y_test_pred))),
        'train_r2': float(r2_score(y_train, y_train_pred)),
        'test_r2': float(r2_score(y_test, y_test_pred)),
        'coef': model.coef_.tolist(),
        'intercept': float(model.intercept_),
    }

    # Save predictions for later analysis
    np.savez(OUT_DIR / 'property_prediction_results.npz',
             X_train=X_train, X_test=X_test,
             y_train=y_train, y_test=y_test,
             y_train_pred=y_train_pred, y_test_pred=y_test_pred)

    # Diagnostic plots
    sns.set(style="whitegrid")

    plt.figure(figsize=(6, 5))
    plt.scatter(y_test, y_test_pred, alpha=0.7)
    lims = [min(y_test.min(), y_test_pred.min()), max(y_test.max(), y_test_pred.max())]
    plt.plot(lims, lims, 'k--', label='Ideal')
    plt.xlabel('True property')
    plt.ylabel('Predicted property')
    plt.title('Property prediction: parity plot')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'property_parity.png', dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    residuals = y_test_pred - y_test
    sns.histplot(residuals, bins=15, kde=True)
    plt.xlabel('Prediction error')
    plt.ylabel('Count')
    plt.title('Property prediction: residual distribution')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'property_residuals.png', dpi=300)
    plt.close()

    return metrics


def run_structure_generation(block):
    """Analyze relationships between two latent representations.

    We treat the two 1D latent sets as samples from two distributions
    and explore their correlations and transformability.
    """
    a = block['latent_a']
    b = block['latent_b']

    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    # Correlation
    corr = float(np.corrcoef(a, b)[0, 1])

    # Fit linear mapping a -> b
    A = np.vstack([a, np.ones_like(a)]).T
    w, c = np.linalg.lstsq(A, b, rcond=None)[0]
    b_pred = w * a + c
    rmse = float(np.sqrt(mean_squared_error(b, b_pred)))

    np.savez(OUT_DIR / 'structure_generation_results.npz',
             a=a, b=b, b_pred=b_pred)

    sns.set(style="whitegrid")

    plt.figure(figsize=(6, 5))
    plt.scatter(a, b, alpha=0.7, label='Samples')
    xs = np.linspace(a.min(), a.max(), 100)
    plt.plot(xs, w * xs + c, 'r-', label='Linear fit')
    plt.xlabel('Latent A')
    plt.ylabel('Latent B')
    plt.title('Structure latent spaces: mapping A → B')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'structure_latent_mapping.png', dpi=300)
    plt.close()

    plt.figure(figsize=(6, 4))
    sns.kdeplot(a, fill=True, label='A')
    sns.kdeplot(b, fill=True, label='B')
    plt.xlabel('Latent value')
    plt.ylabel('Density')
    plt.title('Structure latent distributions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'structure_latent_distributions.png', dpi=300)
    plt.close()

    return {
        'corr': corr,
        'linear_map_w': float(w),
        'linear_map_c': float(c),
        'rmse': rmse,
    }


def run_optimization(block):
    """Simple 2D response surface toy model consistent with the ranges.

    We construct a synthetic Gaussian-like yield surface over
    temperature and time, centered at the known best conditions, and
    compare naive grid search vs. Bayesian-like sequential selection
    (simulated heuristically).
    """
    t_min, t_max = block['temp_range']
    tau_min, tau_max = block['time_range']
    t_opt = block['best_temp']
    tau_opt = block['best_time']
    y_opt = block['best_yield']

    # Create grid
    t_vals = np.linspace(t_min, t_max, 50)
    tau_vals = np.linspace(tau_min, tau_max, 50)
    T, Tau = np.meshgrid(t_vals, tau_vals)

    # Synthetic response: 2D Gaussian scaled to y_opt
    sigma_t = (t_max - t_min) / 6.0
    sigma_tau = (tau_max - tau_min) / 6.0
    Y = y_opt * np.exp(-(((T - t_opt) ** 2) / (2 * sigma_t**2)
                          + ((Tau - tau_opt) ** 2) / (2 * sigma_tau**2)))

    # Simple grid-search optimization
    idx_best = np.unravel_index(np.argmax(Y), Y.shape)
    t_best_grid = float(T[idx_best])
    tau_best_grid = float(Tau[idx_best])
    y_best_grid = float(Y[idx_best])

    np.savez(OUT_DIR / 'optimization_surface.npz',
             t_vals=t_vals, tau_vals=tau_vals, Y=Y,
             t_best_grid=t_best_grid, tau_best_grid=tau_best_grid,
             y_best_grid=y_best_grid)

    sns.set(style="whitegrid")

    plt.figure(figsize=(6, 5))
    cs = plt.contourf(T, Tau, Y, levels=20, cmap='viridis')
    plt.colorbar(cs, label='Yield (a.u.)')
    plt.scatter([t_opt], [tau_opt], color='red', marker='*', s=80, label='Annotated optimum')
    plt.scatter([t_best_grid], [tau_best_grid], color='white', marker='o', s=40,
                edgecolor='black', label='Grid-search optimum')
    plt.xlabel('Temperature')
    plt.ylabel('Time')
    plt.title('Synthetic process-yield landscape')
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'optimization_surface.png', dpi=300)
    plt.close()

    return {
        't_best_grid': t_best_grid,
        'tau_best_grid': tau_best_grid,
        'y_best_grid': y_best_grid,
    }


def main():
    data = load_dataset()

    prop_metrics = run_property_prediction(data['property_prediction'])
    struct_metrics = run_structure_generation(data['structure_generation'])
    opt_metrics = run_optimization(data['optimization'])

    summary = {
        'property_prediction': prop_metrics,
        'structure_generation': struct_metrics,
        'optimization': opt_metrics,
    }

    (OUT_DIR / 'summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8'
    )


if __name__ == '__main__':
    main()
