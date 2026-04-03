"""
Optimal Power Dispatch Model for GB 20-Bus Power System
=======================================================
Linear program minimizing operational cost subject to:
  - Power balance at each bus each hour
  - Generator capacity limits (wind: capacity factor limited)
  - Pumped-hydro storage constraints (energy continuity, power/energy limits)
  - Transmission line capacity limits
"""

import os
import numpy as np
import pandas as pd
import pulp
import json

WORKSPACE = "/mnt/shared-storage-user/yetianlin/ResearchClawBench/workspaces/Energy_001_20260401_191819"
DATA_DIR  = os.path.join(WORKSPACE, "data")
OUT_DIR   = os.path.join(WORKSPACE, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────────────────────────
buses      = pd.read_csv(os.path.join(DATA_DIR, "buses.csv"))
links      = pd.read_csv(os.path.join(DATA_DIR, "links.csv"))
generators = pd.read_csv(os.path.join(DATA_DIR, "generators.csv"))
storage    = pd.read_csv(os.path.join(DATA_DIR, "storage.csv"))
demand_df  = pd.read_csv(os.path.join(DATA_DIR, "demand.csv"), header=0)
wind_cf_df = pd.read_csv(os.path.join(DATA_DIR, "wind_cf.csv"), header=0)

T = len(demand_df)        # 168 hours
bus_names = buses["name"].tolist()
B = len(bus_names)

print(f"Buses: {B},  Hours: {T}")
print(f"Generators: {len(generators)} rows")
print(f"Links: {len(links)} rows")
print(f"Storage units: {len(storage)} rows")

# ──────────────────────────────────────────────────────────────────
# 2. Index helpers
# ──────────────────────────────────────────────────────────────────
bus_idx = {b: i for i, b in enumerate(bus_names)}

# Separate generator types
wind_gens    = generators[generators["carrier"] == "onshore wind"].reset_index(drop=True)
gas_gens     = generators[generators["carrier"] == "gas"].reset_index(drop=True)
nuclear_gens = generators[generators["carrier"] == "nuclear"].reset_index(drop=True)

n_wind    = len(wind_gens)
n_gas     = len(gas_gens)
n_nuclear = len(nuclear_gens)
n_storage = len(storage)
n_links   = len(links)

# ──────────────────────────────────────────────────────────────────
# 3. Build PuLP model
# ──────────────────────────────────────────────────────────────────
model = pulp.LpProblem("GB_Optimal_Dispatch", pulp.LpMinimize)

# ── Decision variables ─────────────────────────────────────────────
# Wind generation  [generator_index, hour]
p_wind = {(g, t): pulp.LpVariable(f"p_wind_{g}_{t}", lowBound=0)
          for g in range(n_wind) for t in range(T)}

# Gas generation
p_gas = {(g, t): pulp.LpVariable(f"p_gas_{g}_{t}", lowBound=0)
         for g in range(n_gas) for t in range(T)}

# Nuclear generation
p_nuc = {(g, t): pulp.LpVariable(f"p_nuc_{g}_{t}", lowBound=0)
         for g in range(n_nuclear) for t in range(T)}

# Storage charge / discharge / state-of-charge
p_ch  = {(s, t): pulp.LpVariable(f"p_ch_{s}_{t}",  lowBound=0)
         for s in range(n_storage) for t in range(T)}
p_dis = {(s, t): pulp.LpVariable(f"p_dis_{s}_{t}", lowBound=0)
         for s in range(n_storage) for t in range(T)}
e_soc = {(s, t): pulp.LpVariable(f"e_soc_{s}_{t}", lowBound=0)
         for s in range(n_storage) for t in range(T)}

# Transmission flow (signed, positive = bus0 → bus1)
f_line = {(l, t): pulp.LpVariable(f"f_{l}_{t}")
          for l in range(n_links) for t in range(T)}

# Load shedding (penalty variable for infeasibility guard)
p_shed = {(b, t): pulp.LpVariable(f"shed_{b}_{t}", lowBound=0)
          for b in range(B) for t in range(T)}

# ── Objective ──────────────────────────────────────────────────────
SHED_COST = 10000  # $/MWh very high to keep shed near zero

model += (
    pulp.lpSum(gas_gens.loc[g, "marginal_cost"] * p_gas[(g, t)]
               for g in range(n_gas) for t in range(T))
  + pulp.lpSum(nuclear_gens.loc[g, "marginal_cost"] * p_nuc[(g, t)]
               for g in range(n_nuclear) for t in range(T))
  + SHED_COST * pulp.lpSum(p_shed[(b, t)]
                            for b in range(B) for t in range(T))
)

# ── Constraints ────────────────────────────────────────────────────

# -- Generator upper bounds --
for g in range(n_wind):
    bus = wind_gens.loc[g, "bus"]
    p_nom = wind_gens.loc[g, "p_nom"]
    for t in range(T):
        cf = wind_cf_df.iloc[t][bus]
        model += p_wind[(g, t)] <= p_nom * cf, f"wind_ub_{g}_{t}"

for g in range(n_gas):
    p_nom = gas_gens.loc[g, "p_nom"]
    for t in range(T):
        model += p_gas[(g, t)] <= p_nom, f"gas_ub_{g}_{t}"

for g in range(n_nuclear):
    p_nom = nuclear_gens.loc[g, "p_nom"]
    for t in range(T):
        model += p_nuc[(g, t)] <= p_nom, f"nuc_ub_{g}_{t}"

# -- Storage power/energy bounds --
for s in range(n_storage):
    p_nom = storage.loc[s, "p_nom"]
    e_nom = storage.loc[s, "e_nom"]
    for t in range(T):
        model += p_ch[(s, t)]  <= p_nom, f"ch_ub_{s}_{t}"
        model += p_dis[(s, t)] <= p_nom, f"dis_ub_{s}_{t}"
        model += e_soc[(s, t)] <= e_nom, f"soc_ub_{s}_{t}"

# -- Storage energy continuity --
for s in range(n_storage):
    eff = storage.loc[s, "efficiency"]
    e_nom = storage.loc[s, "e_nom"]
    for t in range(T):
        if t == 0:
            e_prev = e_nom * 0.5    # initial SoC = 50%
        else:
            e_prev = e_soc[(s, t - 1)]
        model += (
            e_soc[(s, t)] == e_prev
                             + eff * p_ch[(s, t)]
                             - p_dis[(s, t)]
        ), f"soc_cont_{s}_{t}"

# -- Transmission bounds --
for l in range(n_links):
    p_nom = links.loc[l, "p_nom"]
    for t in range(T):
        model += f_line[(l, t)] >=  -p_nom, f"flow_lb_{l}_{t}"
        model += f_line[(l, t)] <=   p_nom, f"flow_ub_{l}_{t}"

# -- Power balance at each bus --
# Build bus-indexed lookup structures for speed
wind_by_bus    = {b: [] for b in range(B)}
gas_by_bus     = {b: [] for b in range(B)}
nuc_by_bus     = {b: [] for b in range(B)}
storage_by_bus = {b: [] for b in range(B)}
line_out       = {b: [] for b in range(B)}   # (link_idx, +1 → outflow)
line_in        = {b: [] for b in range(B)}

for g in range(n_wind):
    b = bus_idx[wind_gens.loc[g, "bus"]]
    wind_by_bus[b].append(g)

for g in range(n_gas):
    b = bus_idx[gas_gens.loc[g, "bus"]]
    gas_by_bus[b].append(g)

for g in range(n_nuclear):
    b = bus_idx[nuclear_gens.loc[g, "bus"]]
    nuc_by_bus[b].append(g)

for s in range(n_storage):
    b = bus_idx[storage.loc[s, "bus"]]
    storage_by_bus[b].append(s)

for l in range(n_links):
    b0 = bus_idx[links.loc[l, "bus0"]]
    b1 = bus_idx[links.loc[l, "bus1"]]
    line_out[b0].append(l)   # flow leaves b0
    line_in[b1].append(l)    # flow enters b1

for b_idx, b_name in enumerate(bus_names):
    for t in range(T):
        d = demand_df.iloc[t][b_name]

        gen_supply = (
            pulp.lpSum(p_wind[(g, t)] for g in wind_by_bus[b_idx])
          + pulp.lpSum(p_gas[(g, t)]  for g in gas_by_bus[b_idx])
          + pulp.lpSum(p_nuc[(g, t)]  for g in nuc_by_bus[b_idx])
          + pulp.lpSum(p_dis[(s, t)]  for s in storage_by_bus[b_idx])
          - pulp.lpSum(p_ch[(s, t)]   for s in storage_by_bus[b_idx])
          + pulp.lpSum(f_line[(l, t)] for l in line_in[b_idx])
          - pulp.lpSum(f_line[(l, t)] for l in line_out[b_idx])
          + p_shed[(b_idx, t)]
        )

        model += gen_supply == d, f"balance_{b_idx}_{t}"

# ──────────────────────────────────────────────────────────────────
# 4. Solve
# ──────────────────────────────────────────────────────────────────
print("Solving LP…")
solver = pulp.PULP_CBC_CMD(msg=1, timeLimit=600)
status = model.solve(solver)
print(f"Status: {pulp.LpStatus[model.status]}")
print(f"Objective (total cost, $): {pulp.value(model.objective):,.0f}")

# ──────────────────────────────────────────────────────────────────
# 5. Extract results
# ──────────────────────────────────────────────────────────────────
def extract_matrix(var_dict, rows, cols):
    """Extract variable values into a numpy array (rows x cols)."""
    arr = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            v = var_dict.get((r, c))
            arr[r, c] = pulp.value(v) if v is not None else 0.0
    return arr

wind_gen_MW    = extract_matrix(p_wind,  n_wind,    T)
gas_gen_MW     = extract_matrix(p_gas,   n_gas,     T)
nuc_gen_MW     = extract_matrix(p_nuc,   n_nuclear, T)
storage_ch_MW  = extract_matrix(p_ch,    n_storage, T)
storage_dis_MW = extract_matrix(p_dis,   n_storage, T)
storage_soc_MWh= extract_matrix(e_soc,   n_storage, T)
flow_MW        = extract_matrix(f_line,  n_links,   T)
shed_MW        = np.zeros((B, T))
for b in range(B):
    for t in range(T):
        v = p_shed.get((b, t))
        shed_MW[b, t] = pulp.value(v) if v is not None else 0.0

# Aggregate generation by type and hour
total_wind_GW    = wind_gen_MW.sum(axis=0)   / 1e3
total_gas_GW     = gas_gen_MW.sum(axis=0)    / 1e3
total_nuclear_GW = nuc_gen_MW.sum(axis=0)    / 1e3
total_storage_net_GW = (storage_dis_MW - storage_ch_MW).sum(axis=0) / 1e3
total_demand_GW  = demand_df.values.sum(axis=1) / 1e3
total_shed_GW    = shed_MW.sum(axis=0)       / 1e3

# Generation by bus
wind_by_bus_GWh  = wind_gen_MW.T.sum(axis=0) / 1e3    # shape: (n_wind,)
gas_by_bus_GWh   = gas_gen_MW.T.sum(axis=0) / 1e3
nuc_by_bus_GWh   = nuc_gen_MW.T.sum(axis=0) / 1e3

# ── Save results ────────────────────────────────────────────────────
np.save(os.path.join(OUT_DIR, "wind_gen_MW.npy"),     wind_gen_MW)
np.save(os.path.join(OUT_DIR, "gas_gen_MW.npy"),      gas_gen_MW)
np.save(os.path.join(OUT_DIR, "nuc_gen_MW.npy"),      nuc_gen_MW)
np.save(os.path.join(OUT_DIR, "storage_ch_MW.npy"),   storage_ch_MW)
np.save(os.path.join(OUT_DIR, "storage_dis_MW.npy"),  storage_dis_MW)
np.save(os.path.join(OUT_DIR, "storage_soc_MWh.npy"), storage_soc_MWh)
np.save(os.path.join(OUT_DIR, "flow_MW.npy"),         flow_MW)
np.save(os.path.join(OUT_DIR, "shed_MW.npy"),         shed_MW)

# Time series DataFrame
ts = pd.DataFrame({
    "hour":         np.arange(T),
    "demand_GW":    total_demand_GW,
    "wind_GW":      total_wind_GW,
    "gas_GW":       total_gas_GW,
    "nuclear_GW":   total_nuclear_GW,
    "storage_net_GW": total_storage_net_GW,
    "shed_GW":      total_shed_GW,
})
ts.to_csv(os.path.join(OUT_DIR, "dispatch_timeseries.csv"), index=False)

# Summary stats
summary = {
    "total_cost_USD":       float(pulp.value(model.objective)),
    "wind_energy_GWh":      float(total_wind_GW.sum()),
    "gas_energy_GWh":       float(total_gas_GW.sum()),
    "nuclear_energy_GWh":   float(total_nuclear_GW.sum()),
    "storage_discharge_GWh":float((storage_dis_MW.sum()) / 1e3),
    "storage_charge_GWh":   float((storage_ch_MW.sum()) / 1e3),
    "load_shed_GWh":        float(total_shed_GW.sum()),
    "wind_share_pct":       float(total_wind_GW.sum() / total_demand_GW.sum() * 100),
    "gas_share_pct":        float(total_gas_GW.sum()  / total_demand_GW.sum() * 100),
    "nuclear_share_pct":    float(total_nuclear_GW.sum() / total_demand_GW.sum() * 100),
}
with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Summary ===")
for k, v in summary.items():
    print(f"  {k}: {v:.2f}")

# Bus-level storage soc summary
storage_info = storage.copy()
storage_info["max_soc_MWh"] = storage_soc_MWh.max(axis=1)
storage_info["mean_soc_MWh"] = storage_soc_MWh.mean(axis=1)
storage_info["total_discharge_GWh"] = storage_dis_MW.sum(axis=1) / 1e3
storage_info["total_charge_GWh"] = storage_ch_MW.sum(axis=1) / 1e3
storage_info.to_csv(os.path.join(OUT_DIR, "storage_summary.csv"), index=False)

print("\nDone. Results saved to", OUT_DIR)
