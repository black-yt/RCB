"""
Generate all figures for the GB dispatch research report.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx
import json

WORKSPACE  = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Energy_001_20260401_191819"
DATA_DIR   = os.path.join(WORKSPACE, "data")
OUT_DIR    = os.path.join(WORKSPACE, "outputs")
IMG_DIR    = os.path.join(WORKSPACE, "report", "images")
os.makedirs(IMG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
})

# ── Load data ─────────────────────────────────────────────────────
buses      = pd.read_csv(os.path.join(DATA_DIR, "buses.csv"))
links      = pd.read_csv(os.path.join(DATA_DIR, "links.csv"))
generators = pd.read_csv(os.path.join(DATA_DIR, "generators.csv"))
storage    = pd.read_csv(os.path.join(DATA_DIR, "storage.csv"))
demand_df  = pd.read_csv(os.path.join(DATA_DIR, "demand.csv"), header=0)
wind_cf_df = pd.read_csv(os.path.join(DATA_DIR, "wind_cf.csv"), header=0)

wind_gen_MW    = np.load(os.path.join(OUT_DIR, "wind_gen_MW.npy"))
gas_gen_MW     = np.load(os.path.join(OUT_DIR, "gas_gen_MW.npy"))
nuc_gen_MW     = np.load(os.path.join(OUT_DIR, "nuc_gen_MW.npy"))
storage_ch_MW  = np.load(os.path.join(OUT_DIR, "storage_ch_MW.npy"))
storage_dis_MW = np.load(os.path.join(OUT_DIR, "storage_dis_MW.npy"))
storage_soc    = np.load(os.path.join(OUT_DIR, "storage_soc_MWh.npy"))
flow_MW        = np.load(os.path.join(OUT_DIR, "flow_MW.npy"))
shed_MW        = np.load(os.path.join(OUT_DIR, "shed_MW.npy"))
ts             = pd.read_csv(os.path.join(OUT_DIR, "dispatch_timeseries.csv"))
with open(os.path.join(OUT_DIR, "summary.json")) as f:
    summary = json.load(f)

T = len(demand_df)
hours = np.arange(T)
bus_names = buses["name"].tolist()
B = len(bus_names)
wind_gens    = generators[generators["carrier"] == "onshore wind"].reset_index(drop=True)
gas_gens     = generators[generators["carrier"] == "gas"].reset_index(drop=True)
nuclear_gens = generators[generators["carrier"] == "nuclear"].reset_index(drop=True)

# ═══════════════════════════════════════════════════════════════════
# Figure 1 — Network topology with generation capacity overlay
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 8))

# Build graph
G = nx.DiGraph()
for _, row in buses.iterrows():
    G.add_node(row["name"], x=row["x"], y=row["y"])
for _, row in links.iterrows():
    G.add_edge(row["bus0"], row["bus1"])

pos = {row["name"]: (row["x"], row["y"]) for _, row in buses.iterrows()}

# Aggregate installed capacity per bus
cap_wind   = generators[generators["carrier"] == "onshore wind"].groupby("bus")["p_nom"].sum()
cap_gas    = generators[generators["carrier"] == "gas"].groupby("bus")["p_nom"].sum()
cap_nuclear= generators[generators["carrier"] == "nuclear"].groupby("bus")["p_nom"].sum()
demand_per_bus = demand_df.mean()   # average hourly demand per bus

# Node sizes proportional to average demand
node_sizes = [demand_per_bus.get(n, 0) / 30 for n in G.nodes()]

# Color nodes by dominant generation type
def node_color(bus):
    has_nuc   = bus in cap_nuclear.index
    wind_cap  = cap_wind.get(bus, 0)
    if has_nuc:
        return "#e41a1c"   # red = nuclear
    elif wind_cap >= 1000:
        return "#377eb8"   # blue = large wind
    else:
        return "#4daf4a"   # green = smaller wind/gas mix

colors = [node_color(n) for n in G.nodes()]

# Draw edges (undirected for display)
nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.6, edge_color="#888888",
                        width=1.5, arrows=False)
# Draw nodes
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=colors,
                        node_size=node_sizes, alpha=0.9)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=7.5, font_color="white",
                         font_weight="bold")

# Legend
patches = [
    mpatches.Patch(color="#e41a1c", label="Nuclear hub"),
    mpatches.Patch(color="#377eb8", label="Large wind (≥1 GW)"),
    mpatches.Patch(color="#4daf4a", label="Gas/mixed"),
]
ax.legend(handles=patches, loc="lower left", fontsize=9, framealpha=0.85)
ax.set_title("GB 20-Bus Power System: Network Topology\n"
             "Node size ∝ mean hourly demand; colour = dominant generation type",
             pad=12)
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig1_network_topology.png"))
plt.close()
print("fig1 saved")

# ═══════════════════════════════════════════════════════════════════
# Figure 2 — Installed capacity & average demand per bus (bar chart)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(B)
width = 0.25

wind_caps  = [cap_wind.get(b, 0) / 1e3 for b in bus_names]
gas_caps   = [cap_gas.get(b, 0)  / 1e3 for b in bus_names]
nuc_caps   = [cap_nuclear.get(b, 0) / 1e3 for b in bus_names]

ax = axes[0]
ax.bar(x - width, wind_caps,  width, label="Wind",    color="#377eb8", alpha=0.85)
ax.bar(x,         gas_caps,   width, label="Gas",     color="#ff7f00", alpha=0.85)
ax.bar(x + width, nuc_caps,   width, label="Nuclear", color="#e41a1c", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([b.replace("Bus", "B") for b in bus_names], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Installed capacity (GW)")
ax.set_title("Installed generation capacity by bus")
ax.legend()

ax = axes[1]
avg_dem = [demand_per_bus[b] / 1e3 for b in bus_names]
max_dem = [demand_df[b].max()  / 1e3 for b in bus_names]
ax.bar(x - 0.2, avg_dem, 0.4, label="Mean demand",  color="#984ea3", alpha=0.85)
ax.bar(x + 0.2, max_dem, 0.4, label="Peak demand",  color="#a65628", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([b.replace("Bus", "B") for b in bus_names], rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Power (GW)")
ax.set_title("Mean and peak demand by bus")
ax.legend()

plt.suptitle("System Capacity and Demand Profile", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig2_capacity_demand.png"), bbox_inches="tight")
plt.close()
print("fig2 saved")

# ═══════════════════════════════════════════════════════════════════
# Figure 3 — Dispatch stack (hourly area chart)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 5))

wind_h    = ts["wind_GW"].values
gas_h     = ts["gas_GW"].values
nuc_h     = ts["nuclear_GW"].values
stor_h    = ts["storage_net_GW"].values
shed_h    = ts["shed_GW"].values
demand_h  = ts["demand_GW"].values

# Positive supply stack
ax.stackplot(hours, wind_h, gas_h, nuc_h,
             labels=["Wind", "Gas", "Nuclear"],
             colors=["#377eb8", "#ff7f00", "#e41a1c"],
             alpha=0.85)

# Storage discharge on top
discharge_h = np.maximum(stor_h, 0)
ax.fill_between(hours, wind_h + gas_h + nuc_h,
                wind_h + gas_h + nuc_h + discharge_h,
                label="Storage (discharge)", color="#4daf4a", alpha=0.75)

# Storage charging as negative
charge_h = np.minimum(stor_h, 0)
ax.fill_between(hours, charge_h, 0, label="Storage (charge)", color="#4daf4a",
                alpha=0.4, hatch="//")

# Demand line
ax.plot(hours, demand_h, color="black", linewidth=1.5, label="Demand", zorder=5)

# Shed region
ax.fill_between(hours, demand_h - shed_h, demand_h,
                color="darkred", alpha=0.35, label="Load shed", zorder=4)

ax.set_xlabel("Hour")
ax.set_ylabel("Power (GW)")
ax.set_title("Optimal Dispatch Stack — 168-hour Simulation (One Week)")
ax.legend(loc="upper right", ncol=3, fontsize=9)
ax.set_xlim(0, T - 1)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig3_dispatch_stack.png"))
plt.close()
print("fig3 saved")

# ═══════════════════════════════════════════════════════════════════
# Figure 4 — Wind capacity factor heatmap (bus × hour)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 5))
cf_matrix = wind_cf_df.values.T   # shape: (20 buses, 168 hours)
sns.heatmap(cf_matrix, ax=ax,
            xticklabels=24,
            yticklabels=[b.replace("Bus", "B") for b in bus_names],
            cmap="YlOrRd", vmin=0, vmax=1,
            cbar_kws={"label": "Capacity factor"},
            linewidths=0)
ax.set_xlabel("Hour")
ax.set_ylabel("Bus")
ax.set_title("Wind Capacity Factor — Spatial and Temporal Variability")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig4_wind_cf_heatmap.png"))
plt.close()
print("fig4 saved")

# ═══════════════════════════════════════════════════════════════════
# Figure 5 — Generation mix pie chart (total GWh)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

total_served = (summary["wind_energy_GWh"] + summary["gas_energy_GWh"] +
                summary["nuclear_energy_GWh"] + summary["storage_discharge_GWh"])
total_demand = total_served + summary["load_shed_GWh"]

# Left: Generation mix (of what was served)
values_gen = [summary["wind_energy_GWh"], summary["gas_energy_GWh"],
              summary["nuclear_energy_GWh"], summary["storage_discharge_GWh"]]
labels_gen = ["Wind", "Gas", "Nuclear", "Storage"]
colors_gen = ["#377eb8", "#ff7f00", "#e41a1c", "#4daf4a"]
wedges, texts, autotexts = axes[0].pie(values_gen, labels=labels_gen, colors=colors_gen,
                                        autopct="%1.1f%%", startangle=90,
                                        pctdistance=0.8)
axes[0].set_title(f"Generation Mix (Served = {total_served:.0f} GWh)")

# Right: Served vs shed
values_bal = [total_served, summary["load_shed_GWh"]]
labels_bal = [f"Served ({total_served / total_demand * 100:.1f}%)",
              f"Shed ({summary['load_shed_GWh'] / total_demand * 100:.1f}%)"]
colors_bal = ["#2ca25f", "#d73027"]
axes[1].pie(values_bal, labels=labels_bal, colors=colors_bal,
            autopct="%1.1f%%", startangle=90, pctdistance=0.7)
axes[1].set_title(f"Supply Adequacy (Total Demand = {total_demand:.0f} GWh)")

plt.suptitle("Energy Balance and Generation Mix — Full Week", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig5_generation_mix.png"))
plt.close()
print("fig5 saved")

# ═══════════════════════════════════════════════════════════════════
# Figure 6 — Transmission flows (line loading heatmap)
# ═══════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 6))
# Normalise by link capacity
line_caps = links["p_nom"].values   # shape: (n_links,)
loading = np.abs(flow_MW) / line_caps[:, np.newaxis]  # (n_links, T)

link_labels = [f"{r['bus0'][-2:]}-{r['bus1'][-2:]}" for _, r in links.iterrows()]
sns.heatmap(loading, ax=ax,
            xticklabels=24,
            yticklabels=link_labels,
            cmap="coolwarm", vmin=0, vmax=1,
            cbar_kws={"label": "Line loading fraction (|flow| / capacity)"},
            linewidths=0)
ax.set_xlabel("Hour")
ax.set_ylabel("Transmission Link")
ax.set_title("Transmission Line Loading — Fraction of Rated Capacity")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig6_line_loading.png"))
plt.close()
print("fig6 saved")

# ═══════════════════════════════════════════════════════════════════
# Figure 7 — Storage state-of-charge (all 3 units)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
colors_s = ["#1b9e77", "#d95f02", "#7570b3"]

for s in range(3):
    ax = axes[s]
    e_nom = storage.loc[s, "e_nom"]
    soc_pct = storage_soc[s] / e_nom * 100
    charge_pct  = storage_ch_MW[s]  / storage.loc[s, "p_nom"]
    discharge_pct = storage_dis_MW[s] / storage.loc[s, "p_nom"]

    ax.fill_between(hours, soc_pct, alpha=0.45, color=colors_s[s], label="SoC (%)")
    ax.plot(hours, soc_pct, color=colors_s[s], linewidth=1.5)
    ax2 = ax.twinx()
    ax2.bar(hours, discharge_pct * 100, color="#2ca25f", alpha=0.6, width=0.9,
            label="Discharge %")
    ax2.bar(hours, -charge_pct * 100,  color="#d73027", alpha=0.6, width=0.9,
            label="Charge %")
    ax2.set_ylabel("Power (% p_nom)", fontsize=8)
    ax2.set_ylim(-120, 120)

    bus_name = storage.loc[s, "bus"]
    e_nom_MWh = storage.loc[s, "e_nom"]
    p_nom_MW  = storage.loc[s, "p_nom"]
    ax.set_ylabel("SoC (%)")
    ax.set_ylim(0, 110)
    ax.set_title(f"PHS at {bus_name} | {p_nom_MW} MW / {e_nom_MWh} MWh", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)

axes[-1].set_xlabel("Hour")
plt.suptitle("Pumped-Hydro Storage — State of Charge and Power", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig7_storage_soc.png"))
plt.close()
print("fig7 saved")

# ═══════════════════════════════════════════════════════════════════
# Figure 8 — Load shedding by bus (spatial)
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

total_shed_bus  = shed_MW.sum(axis=1) / 1e3   # GWh per bus
total_dem_bus   = demand_df.values.sum(axis=0) / 1e3
shed_frac_bus   = total_shed_bus / np.maximum(total_dem_bus, 1e-9) * 100

# Bar chart of shed by bus
ax = axes[0]
colors_shed = ["#d73027" if f > 50 else "#fc8d59" if f > 25 else "#fee090"
               for f in shed_frac_bus]
ax.bar(range(B), shed_frac_bus, color=colors_shed, alpha=0.9)
ax.set_xticks(range(B))
ax.set_xticklabels([b.replace("Bus", "B") for b in bus_names],
                   rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Load shed (%)")
ax.set_title("Load Shedding Fraction per Bus")
ax.axhline(shed_frac_bus.mean(), color="black", linestyle="--",
           linewidth=1.5, label=f"Average = {shed_frac_bus.mean():.1f}%")
ax.legend()

# Geographic scatter of shed fraction
ax = axes[1]
sc = ax.scatter(buses["x"], buses["y"],
                c=shed_frac_bus, cmap="RdYlGn_r", s=200, alpha=0.9,
                vmin=0, vmax=100, edgecolors="grey", linewidth=0.5)
# Draw links
for _, row in links.iterrows():
    b0 = buses[buses["name"] == row["bus0"]].iloc[0]
    b1 = buses[buses["name"] == row["bus1"]].iloc[0]
    ax.plot([b0["x"], b1["x"]], [b0["y"], b1["y"]], "grey", alpha=0.5, linewidth=1)
for i, row in buses.iterrows():
    ax.annotate(row["name"].replace("Bus", "B"), (row["x"], row["y"]),
                fontsize=6.5, ha="center", va="bottom")
plt.colorbar(sc, ax=ax, label="Load shed (%)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.set_title("Spatial Distribution of Load Shedding")
ax.grid(True, linestyle="--", alpha=0.3)

plt.suptitle("Load Shedding Analysis", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig8_load_shedding.png"), bbox_inches="tight")
plt.close()
print("fig8 saved")

# ═══════════════════════════════════════════════════════════════════
# Figure 9 — Marginal cost comparison & generation by carrier
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Utilization by bus for gas generators
gas_util = gas_gen_MW.sum(axis=1) / (gas_gens["p_nom"].values * T) * 100
ax = axes[0]
ax.bar(range(len(gas_gens)), gas_util, color="#ff7f00", alpha=0.85)
ax.set_xticks(range(len(gas_gens)))
ax.set_xticklabels([b.replace("Bus", "B") for b in gas_gens["bus"].tolist()],
                   rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Capacity utilisation (%)")
ax.set_title("Gas Generator Utilisation by Bus")
ax.axhline(gas_util.mean(), color="black", linestyle="--",
           linewidth=1.5, label=f"Mean = {gas_util.mean():.1f}%")
ax.legend()

# Wind utilisation
wind_util = wind_gen_MW.sum(axis=1) / (wind_gens["p_nom"].values * T) * 100
ax = axes[1]
ax.bar(range(len(wind_gens)), wind_util, color="#377eb8", alpha=0.85)
ax.set_xticks(range(len(wind_gens)))
ax.set_xticklabels([b.replace("Bus", "B") for b in wind_gens["bus"].tolist()],
                   rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Capacity utilisation (%)")
ax.set_title("Wind Generator Utilisation by Bus\n(relative to rated capacity × hours)")
ax.axhline(wind_util.mean(), color="black", linestyle="--",
           linewidth=1.5, label=f"Mean = {wind_util.mean():.1f}%")
ax.legend()

plt.suptitle("Generator Utilisation Rates", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig9_generator_utilisation.png"), bbox_inches="tight")
plt.close()
print("fig9 saved")

# ═══════════════════════════════════════════════════════════════════
# Figure 10 — Demand time series + renewable fraction
# ═══════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

ax = axes[0]
ax.plot(hours, demand_h, color="black", linewidth=1.4, label="Total demand")
ax.fill_between(hours, demand_h, alpha=0.1, color="black")
ax.set_ylabel("Power (GW)")
ax.set_title("System Demand Profile")
ax.legend()

ax = axes[1]
ren_fraction = np.where(demand_h > 0, wind_h / demand_h * 100, 0)
ax.plot(hours, ren_fraction, color="#377eb8", linewidth=1.5, label="Wind penetration")
ax.axhline(ren_fraction.mean(), color="#377eb8", linestyle="--",
           linewidth=1.2, label=f"Mean = {ren_fraction.mean():.1f}%")
ax.fill_between(hours, ren_fraction, alpha=0.25, color="#377eb8")
ax.set_ylabel("Wind / Demand (%)")
ax.set_xlabel("Hour")
ax.set_title("Instantaneous Wind Penetration Rate")
ax.legend()

plt.suptitle("Demand and Renewable Penetration", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "fig10_demand_renewable.png"))
plt.close()
print("fig10 saved")

print("\nAll figures saved to", IMG_DIR)
