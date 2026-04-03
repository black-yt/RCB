import numpy as np
from netCDF4 import Dataset
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'outputs'
FIG_DIR = BASE_DIR / 'report' / 'images'

FIG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set(style='whitegrid', context='talk')


def load_datasets():
    inp_path = DATA_DIR / '20231012-06_input_netcdf.nc'
    fx_path = DATA_DIR / '006.nc'

    ds_in = Dataset(inp_path)
    ds_fx = Dataset(fx_path)

    data_in = ds_in.variables['data'][:]  # (time=2, level=70, lat=181, lon=360)
    data_fx = ds_fx.variables['data'][:]  # (time=1, step=1, level=70, lat=181, lon=360)

    lats = ds_in.variables['lat'][:]
    lons = ds_in.variables['lon'][:]
    levels = ds_in.variables['level'][:]  # encoded meta; treat index as channel

    ds_in.close()
    ds_fx.close()

    return data_in, data_fx, lats, lons, levels


def save_np_stats(array: np.ndarray, name: str):
    arr = array.astype(float)
    stats = {
        'name': name,
        'shape': arr.shape,
        'min': float(np.nanmin(arr)),
        'max': float(np.nanmax(arr)),
        'mean': float(np.nanmean(arr)),
        'std': float(np.nanstd(arr)),
    }
    out_path = OUTPUT_DIR / f'stats_{name}.txt'
    with out_path.open('w') as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
    print('Wrote stats to', out_path)


def plot_global_map(field2d: np.ndarray, lats: np.ndarray, lons: np.ndarray, title: str, fname: str, cmap='coolwarm', vmin=None, vmax=None):
    plt.figure(figsize=(12, 5))
    # assume lats (181,), lons (360,). Use pcolormesh-style grid
    Lon, Lat = np.meshgrid(lons, lats)
    m = plt.pcolormesh(Lon, Lat, field2d, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(m, label='value (normalized)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(title)
    plt.tight_layout()
    out_path = FIG_DIR / fname
    plt.savefig(out_path, dpi=150)
    plt.close()
    print('Saved figure', out_path)


def main():
    data_in, data_fx, lats, lons, levels = load_datasets()

    # Basic stats
    save_np_stats(data_in, 'input')
    save_np_stats(data_fx, 'fuxi')

    # Select a mid-tropospheric level/channel (index 30 arbitrarily)
    level_idx = 30

    # Use last input time as t0 and FuXi forecast at its lead (single step available)
    t_in = 1
    t_fx = 0

    field_in = data_in[t_in, level_idx]
    field_fx = data_fx[t_fx, 0, level_idx]

    # Data overview plots
    vmin = min(field_in.min(), field_fx.min())
    vmax = max(field_in.max(), field_fx.max())

    plot_global_map(field_in, lats, lons, title=f'Input state level {level_idx}', fname='data_overview_input_level30.png', vmin=vmin, vmax=vmax)
    plot_global_map(field_fx, lats, lons, title=f'FuXi forecast level {level_idx}', fname='data_overview_fuxi_level30.png', vmin=vmin, vmax=vmax)

    # Difference map
    diff = field_fx - field_in
    plot_global_map(diff, lats, lons, title=f'FuXi - Input difference level {level_idx}', fname='comparison_diff_level30.png', cmap='bwr')

    # Zonal mean comparison
    zonal_in = field_in.mean(axis=1)
    zonal_fx = field_fx.mean(axis=1)

    plt.figure(figsize=(10, 5))
    plt.plot(lats, zonal_in, label='Initial state')
    plt.plot(lats, zonal_fx, label='FuXi forecast')
    plt.xlabel('Latitude')
    plt.ylabel('Zonal-mean value (normalized)')
    plt.title(f'Zonal-mean comparison at level {level_idx}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = FIG_DIR / 'zonal_mean_comparison_level30.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print('Saved figure', out_path)

    # Simple vertical profile of global-mean variance as a crude scale diagnostic
    var_in_levels = data_in[t_in].reshape(70, -1).var(axis=1)
    var_fx_levels = data_fx[t_fx, 0].reshape(70, -1).var(axis=1)

    plt.figure(figsize=(6, 6))
    plt.plot(var_in_levels, np.arange(70), label='Initial state')
    plt.plot(var_fx_levels, np.arange(70), label='FuXi forecast')
    plt.gca().invert_yaxis()
    plt.xlabel('Global variance (normalized units^2)')
    plt.ylabel('Channel index')
    plt.title('Vertical-channel variance structure')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = FIG_DIR / 'vertical_variance_structure.png'
    plt.savefig(out_path, dpi=150)
    plt.close()
    print('Saved figure', out_path)


if __name__ == '__main__':
    main()
