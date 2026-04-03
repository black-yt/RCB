import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_data_overview(data_path="data", fig_path="report/images"):
    fig_path = Path(fig_path)
    fig_path.mkdir(parents=True, exist_ok=True)

    demand = pd.read_csv(f"{data_path}/demand.csv")
    demand.index.name = "hour"

    plt.figure(figsize=(10, 4))
    demand.sum(axis=1).plot()
    plt.ylabel("Total demand [MW]")
    plt.xlabel("Hour")
    plt.title("System-wide demand over time")
    plt.tight_layout()
    plt.savefig(fig_path / "demand_total.png")
    plt.close()

    wind_cf = pd.read_csv(f"{data_path}/wind_cf.csv")
    plt.figure(figsize=(10, 4))
    wind_cf.mean(axis=1).plot()
    plt.ylabel("Average wind capacity factor")
    plt.xlabel("Hour")
    plt.title("Average wind resource over time")
    plt.tight_layout()
    plt.savefig(fig_path / "wind_cf_avg.png")
    plt.close()


def plot_dispatch(outputs_path="outputs", fig_path="report/images"):
    fig_path = Path(fig_path)
    fig_path.mkdir(parents=True, exist_ok=True)

    gen_p = pd.read_csv(f"{outputs_path}/generators_p.csv", index_col=0, parse_dates=True)

    # split by carrier name encoded in index
    carriers = {}
    for col in gen_p.columns:
        carrier = col.split("_")[0]
        carriers.setdefault(carrier, []).append(col)

    carrier_p = {c: gen_p[cols].sum(axis=1) for c, cols in carriers.items()}
    carrier_df = pd.DataFrame(carrier_p)

    plt.figure(figsize=(10, 5))
    carrier_df.plot.area()
    plt.ylabel("Generation [MW]")
    plt.xlabel("Time")
    plt.title("System-wide generation by carrier")
    plt.tight_layout()
    plt.savefig(fig_path / "dispatch_by_carrier.png")
    plt.close()


if __name__ == "__main__":
    plot_data_overview()
    plot_dispatch()
