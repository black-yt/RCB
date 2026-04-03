import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

BASE = Path(__file__).resolve().parents[1]
DATA_DIR = BASE / 'data'
OUT_DIR = BASE / 'outputs'
FIG_DIR = BASE / 'report' / 'images'
OUT_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)

INIT_FILE = DATA_DIR / '184_verified_Original Data_ML_20230926.xlsx'
OPT_FILE1 = DATA_DIR / 'ML_ei&pred (1&2&3rounds)_20240408.xlsx'
OPT_FILE2 = DATA_DIR / 'ML_ei&pred_20240213.xlsx'

FEATURE_COLS = ['Nucleophilic-HEA', 'Hydrophobic-BA', 'Acidic-CBEA',
                 'Cationic-ATAC', 'Aromatic-PEA', 'Amide-AAm']
TARGET_COL_INIT = 'Glass (kPa)_10s'
TARGET_COL_OPT = 'Glass (kPa)_max'


def load_datasets():
    init = pd.read_excel(INIT_FILE)
    opt1 = pd.read_excel(OPT_FILE1)
    opt2 = pd.read_excel(OPT_FILE2)
    opt = pd.concat([opt1, opt2], ignore_index=True)
    return init, opt


def preprocess(df, feature_cols, target_col):
    df = df.copy()
    df = df[feature_cols + [target_col]].copy()
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df = df.dropna(subset=feature_cols + [target_col])
    X = df[feature_cols].values.astype(float)
    y = df[target_col].values.astype(float)
    return X, y, df


def plot_data_overview(init_df, opt_df):
    # distribution of target
    plt.figure(figsize=(6,4))
    sns.histplot(init_df[TARGET_COL_INIT].dropna(), kde=True)
    plt.xlabel('Adhesive strength on glass at 60 s (kPa)')
    plt.title('Initial dataset: target distribution')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'initial_target_distribution.png', dpi=300)
    plt.close()

    opt_target = pd.to_numeric(opt_df[TARGET_COL_OPT], errors='coerce')
    plt.figure(figsize=(6,4))
    sns.histplot(opt_target.dropna(), kde=True)
    plt.xlabel('Adhesive strength on glass (max, kPa)')
    plt.title('Optimization rounds: target distribution')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'opt_target_distribution.png', dpi=300)
    plt.close()

    # pairplot of features vs target (initial)
    cols = FEATURE_COLS + [TARGET_COL_INIT]
    g = sns.pairplot(init_df[cols].dropna(), diag_kind='hist')
    g.fig.suptitle('Initial dataset: features vs target', y=1.02)
    g.savefig(FIG_DIR / 'initial_pairplot.png', dpi=300)
    plt.close('all')


def train_base_model(X, y, random_state=0):
    model = RandomForestRegressor(n_estimators=300,
                                  max_depth=None,
                                  random_state=random_state,
                                  n_jobs=-1)
    model.fit(X, y)
    return model


def evaluate_model(model, X, y, name_prefix):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    # parity plot
    plt.figure(figsize=(5,5))
    plt.scatter(y_test, preds, alpha=0.7)
    lims = [min(y_test.min(), preds.min()), max(y_test.max(), preds.max())]
    plt.plot(lims, lims, 'k--', linewidth=1)
    plt.xlabel('Measured adhesive strength (kPa)')
    plt.ylabel('Predicted adhesive strength (kPa)')
    plt.title(f'{name_prefix} model parity plot')
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'{name_prefix}_parity.png', dpi=300)
    plt.close()

    # feature importance
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(6,4))
    sns.barplot(x=np.array(FEATURE_COLS)[order], y=importances[order])
    plt.ylabel('Importance')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{name_prefix} feature importance (RF)')
    plt.tight_layout()
    plt.savefig(FIG_DIR / f'{name_prefix}_feature_importance.png', dpi=300)
    plt.close()

    metrics = {
        'r2_test': r2,
        'rmse_test': rmse,
        'r2_cv_mean': float(cv_scores.mean()),
        'r2_cv_std': float(cv_scores.std()),
    }
    return metrics


def analyze_optimization_trajectories(model, opt_df):
    df = opt_df.copy()
    for c in FEATURE_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df[TARGET_COL_OPT] = pd.to_numeric(df[TARGET_COL_OPT], errors='coerce')
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL_OPT])
    X_opt = df[FEATURE_COLS].values.astype(float)
    y_opt = df[TARGET_COL_OPT].values.astype(float)
    y_pred = model.predict(X_opt)
    df['pred_RF'] = y_pred

    # assuming 'ML' encodes design iteration
    if 'ML' in df.columns:
        # sort by ML and look at best-so-far curve
        df_sorted = df.sort_values('ML')
        best_so_far = df_sorted[TARGET_COL_OPT].cummax()
        plt.figure(figsize=(6,4))
        plt.plot(range(len(df_sorted)), best_so_far.values, marker='o', linestyle='-')
        plt.xlabel('Design index (ML)')
        plt.ylabel('Best adhesive strength so far (kPa)')
        plt.title('Optimization trajectory of best adhesive strength')
        plt.tight_layout()
        plt.savefig(FIG_DIR / 'optimization_trajectory_best.png', dpi=300)
        plt.close()

    # calibration on optimization data
    plt.figure(figsize=(5,5))
    plt.scatter(y_opt, y_pred, alpha=0.7)
    lims = [min(y_opt.min(), y_pred.min()), max(y_opt.max(), y_pred.max())]
    plt.plot(lims, lims, 'k--', linewidth=1)
    plt.xlabel('Measured adhesive strength (kPa)')
    plt.ylabel('RF predicted adhesive strength (kPa)')
    plt.title('Calibration on optimization dataset')
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'opt_calibration_parity.png', dpi=300)
    plt.close()

    # save annotated optimization data
    df.to_csv(OUT_DIR / 'optimization_with_predictions.csv', index=False)



def main():
    init_df, opt_df = load_datasets()

    # overview plots
    plot_data_overview(init_df, opt_df)

    # base model on initial dataset
    X_init, y_init, init_used = preprocess(init_df, FEATURE_COLS, TARGET_COL_INIT)
    base_model = train_base_model(X_init, y_init)
    base_metrics = evaluate_model(base_model, X_init, y_init, 'initial_RF')

    # analyze optimization rounds with base model
    analyze_optimization_trajectories(base_model, opt_df)

    # save metrics
    metrics_df = pd.DataFrame([base_metrics])
    metrics_df.to_csv(OUT_DIR / 'model_metrics.csv', index=False)


if __name__ == '__main__':
    main()
