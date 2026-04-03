from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
IMAGE_DIR = ROOT / "report" / "images"


LOAD_SHEDDING_COST = 10000.0
CURTAILMENT_COST = 0.0
INITIAL_SOC_FRACTION = 0.5
ROUNDTRIP_EFFICIENCY_SPLIT = "sqrt"


plt.style.use("seaborn-v0_8-whitegrid")


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> dict[str, pd.DataFrame]:
    return {
        "buses": pd.read_csv(DATA_DIR / "buses.csv"),
        "links": pd.read_csv(DATA_DIR / "links.csv"),
        "generators": pd.read_csv(DATA_DIR / "generators.csv"),
        "storage": pd.read_csv(DATA_DIR / "storage.csv"),
        "demand": pd.read_csv(DATA_DIR / "demand.csv"),
        "wind_cf": pd.read_csv(DATA_DIR / "wind_cf.csv"),
    }


def validate_data(data: dict[str, pd.DataFrame]) -> dict[str, object]:
    buses = data["buses"].copy()
    links = data["links"].copy()
    generators = data["generators"].copy()
    storage = data["storage"].copy()
    demand = data["demand"].copy()
    wind_cf = data["wind_cf"].copy()

    bus_names = set(buses["name"])
    issues: list[str] = []

    for col in ["bus0", "bus1"]:
        missing = sorted(set(links[col]) - bus_names)
        if missing:
            issues.append(f"Unknown buses in links.{col}: {missing}")

    for col in ["bus"]:
        missing_g = sorted(set(generators[col]) - bus_names)
        missing_s = sorted(set(storage[col]) - bus_names)
        if missing_g:
            issues.append(f"Unknown buses in generators.{col}: {missing_g}")
        if missing_s:
            issues.append(f"Unknown buses in storage.{col}: {missing_s}")

    demand_missing = sorted(set(demand.columns) - bus_names)
    wind_missing = sorted(set(wind_cf.columns) - bus_names)
    demand_absent = sorted(bus_names - set(demand.columns))
    wind_absent = sorted(bus_names - set(wind_cf.columns))

    if demand_missing:
        issues.append(f"Demand columns not in buses.csv: {demand_missing}")
    if wind_missing:
        issues.append(f"Wind columns not in buses.csv: {wind_missing}")
    if demand_absent:
        issues.append(f"Buses missing from demand.csv: {demand_absent}")
    if wind_absent:
        issues.append(f"Buses missing from wind_cf.csv: {wind_absent}")
    if len(demand) != len(wind_cf):
        issues.append("Demand and wind_cf have different numbers of hourly snapshots")

    summary = {
        "n_buses": int(len(buses)),
        "n_links": int(len(links)),
        "n_generators": int(len(generators)),
        "n_storage_units": int(len(storage)),
        "n_hours": int(len(demand)),
        "generator_capacity_mw_by_carrier": generators.groupby("carrier")["p_nom"].sum().round(3).to_dict(),
        "storage_power_mw": float(storage["p_nom"].sum()),
        "storage_energy_mwh": float(storage["e_nom"].sum()),
        "total_demand_mwh": float(demand.sum().sum()),
        "peak_system_demand_mw": float(demand.sum(axis=1).max()),
        "issues": issues,
    }
    return summary


def build_and_solve_model(data: dict[str, pd.DataFrame]) -> dict[str, object]:
    buses = data["buses"].copy()
    links = data["links"].copy().reset_index(drop=True)
    generators = data["generators"].copy().reset_index(drop=True)
    storage = data["storage"].copy().reset_index(drop=True)
    demand = data["demand"].copy()
    wind_cf = data["wind_cf"].copy()

    bus_order = buses["name"].tolist()
    bus_to_idx = {b: i for i, b in enumerate(bus_order)}
    snapshots = np.arange(len(demand))
    hours = len(snapshots)

    if ROUNDTRIP_EFFICIENCY_SPLIT == "sqrt":
        storage["eta_charge"] = np.sqrt(storage["efficiency"])
        storage["eta_discharge"] = np.sqrt(storage["efficiency"])
    else:
        storage["eta_charge"] = storage["efficiency"]
        storage["eta_discharge"] = 1.0

    gen_groups = generators.groupby("carrier").groups
    wind_idx = list(gen_groups.get("onshore wind", []))
    nonwind_idx = [i for i in generators.index if i not in wind_idx]

    demand_matrix = demand[bus_order].to_numpy(dtype=float)

    wind_availability = np.zeros((hours, len(generators)), dtype=float)
    for g, row in generators.iterrows():
        if row["carrier"] == "onshore wind":
            wind_availability[:, g] = wind_cf[row["bus"]].to_numpy(dtype=float) * float(row["p_nom"])
        else:
            wind_availability[:, g] = float(row["p_nom"])

    n_g = len(generators)
    n_s = len(storage)
    n_l = len(links)
    n_b = len(buses)

    n_pg = hours * n_g
    n_charge = hours * n_s
    n_discharge = hours * n_s
    n_soc = hours * n_s
    n_flow = hours * n_l
    n_shed = hours * n_b
    n_vars = n_pg + n_charge + n_discharge + n_soc + n_flow + n_shed

    offset_pg = 0
    offset_charge = offset_pg + n_pg
    offset_discharge = offset_charge + n_charge
    offset_soc = offset_discharge + n_discharge
    offset_flow = offset_soc + n_soc
    offset_shed = offset_flow + n_flow

    def vidx_pg(t: int, g: int) -> int:
        return offset_pg + t * n_g + g

    def vidx_charge(t: int, s: int) -> int:
        return offset_charge + t * n_s + s

    def vidx_discharge(t: int, s: int) -> int:
        return offset_discharge + t * n_s + s

    def vidx_soc(t: int, s: int) -> int:
        return offset_soc + t * n_s + s

    def vidx_flow(t: int, l: int) -> int:
        return offset_flow + t * n_l + l

    def vidx_shed(t: int, b: int) -> int:
        return offset_shed + t * n_b + b

    c = np.zeros(n_vars, dtype=float)
    for t in snapshots:
        for g, row in generators.iterrows():
            c[vidx_pg(t, g)] = float(row["marginal_cost"])
        for b in range(n_b):
            c[vidx_shed(t, b)] = LOAD_SHEDDING_COST

    bounds: list[tuple[float | None, float | None]] = [(0.0, None)] * n_vars

    for t in snapshots:
        for g in generators.index:
            bounds[vidx_pg(t, g)] = (0.0, float(wind_availability[t, g]))
        for s, row in storage.iterrows():
            bounds[vidx_charge(t, s)] = (0.0, float(row["p_nom"]))
            bounds[vidx_discharge(t, s)] = (0.0, float(row["p_nom"]))
            bounds[vidx_soc(t, s)] = (0.0, float(row["e_nom"]))
        for l, row in links.iterrows():
            cap = float(row["p_nom"])
            bounds[vidx_flow(t, l)] = (-cap, cap)
        for b in range(n_b):
            bounds[vidx_shed(t, b)] = (0.0, float(demand_matrix[t, b]))

    eq_rows = []
    eq_cols = []
    eq_vals = []
    b_eq = []
    row_counter = 0

    gens_by_bus = generators.groupby("bus").groups
    storage_by_bus = storage.groupby("bus").groups
    outgoing_by_bus: dict[str, list[int]] = {b: [] for b in bus_order}
    incoming_by_bus: dict[str, list[int]] = {b: [] for b in bus_order}
    for l, row in links.iterrows():
        outgoing_by_bus[row["bus0"]].append(l)
        incoming_by_bus[row["bus1"]].append(l)

    # Nodal balances per hour
    for t in snapshots:
        for b, bus in enumerate(bus_order):
            for g in gens_by_bus.get(bus, []):
                eq_rows.append(row_counter)
                eq_cols.append(vidx_pg(t, g))
                eq_vals.append(1.0)
            for s in storage_by_bus.get(bus, []):
                eq_rows.extend([row_counter, row_counter])
                eq_cols.extend([vidx_discharge(t, s), vidx_charge(t, s)])
                eq_vals.extend([1.0, -1.0])
            eq_rows.append(row_counter)
            eq_cols.append(vidx_shed(t, b))
            eq_vals.append(1.0)
            for l in incoming_by_bus[bus]:
                eq_rows.append(row_counter)
                eq_cols.append(vidx_flow(t, l))
                eq_vals.append(1.0)
            for l in outgoing_by_bus[bus]:
                eq_rows.append(row_counter)
                eq_cols.append(vidx_flow(t, l))
                eq_vals.append(-1.0)
            b_eq.append(float(demand_matrix[t, b]))
            row_counter += 1

    # Storage state transition constraints
    initial_soc = (storage["e_nom"] * INITIAL_SOC_FRACTION).to_numpy(dtype=float)
    for s, row in storage.iterrows():
        eta_c = float(row["eta_charge"])
        eta_d = float(row["eta_discharge"])
        for t in snapshots:
            eq_rows.extend([row_counter, row_counter, row_counter])
            eq_cols.extend([vidx_soc(t, s), vidx_charge(t, s), vidx_discharge(t, s)])
            eq_vals.extend([1.0, -eta_c, 1.0 / eta_d])
            if t > 0:
                eq_rows.append(row_counter)
                eq_cols.append(vidx_soc(t - 1, s))
                eq_vals.append(-1.0)
                rhs = 0.0
            else:
                # soc_0 = initial_soc + eta_c*charge_0 - discharge_0/eta_d
                rhs = float(initial_soc[s])
            b_eq.append(rhs)
            row_counter += 1

        # End-of-horizon SOC equals initial SOC for cyclical consistency
        eq_rows.append(row_counter)
        eq_cols.append(vidx_soc(hours - 1, s))
        eq_vals.append(1.0)
        b_eq.append(float(initial_soc[s]))
        row_counter += 1

    A_eq = coo_matrix((eq_vals, (eq_rows, eq_cols)), shape=(row_counter, n_vars)).tocsr()
    b_eq = np.array(b_eq, dtype=float)

    result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    x = result.x

    dispatch = np.array([x[offset_pg:offset_charge]]).reshape(hours, n_g)
    charge = np.array([x[offset_charge:offset_discharge]]).reshape(hours, n_s)
    discharge = np.array([x[offset_discharge:offset_soc]]).reshape(hours, n_s)
    soc = np.array([x[offset_soc:offset_flow]]).reshape(hours, n_s)
    flow = np.array([x[offset_flow:offset_shed]]).reshape(hours, n_l)
    shed = np.array([x[offset_shed:]]).reshape(hours, n_b)

    dispatch_df = pd.DataFrame(dispatch, columns=[f"g{i}" for i in generators.index])
    dispatch_long = dispatch_df.T
    dispatch_long["bus"] = generators["bus"].values
    dispatch_long["carrier"] = generators["carrier"].values
    dispatch_by_carrier = dispatch_long.groupby("carrier").sum(numeric_only=True).T
    dispatch_by_carrier.index.name = "hour"

    generator_dispatch = pd.DataFrame(dispatch, columns=generators.index)
    wind_available_by_gen = pd.DataFrame(wind_availability, columns=generators.index)
    wind_generation = generator_dispatch[wind_idx].copy() if wind_idx else pd.DataFrame(index=range(hours))
    wind_available = wind_available_by_gen[wind_idx].copy() if wind_idx else pd.DataFrame(index=range(hours))
    wind_curtailment = wind_available - wind_generation if wind_idx else pd.DataFrame(index=range(hours))

    storage_charge_df = pd.DataFrame(charge, columns=storage.index)
    storage_discharge_df = pd.DataFrame(discharge, columns=storage.index)
    storage_soc_df = pd.DataFrame(soc, columns=storage.index)
    flow_df = pd.DataFrame(flow, columns=links.index)
    shed_df = pd.DataFrame(shed, columns=bus_order)

    objective = float(result.fun)
    generation_cost = float((dispatch * generators["marginal_cost"].to_numpy(dtype=float)).sum())
    load_shedding_cost = float(shed.sum() * LOAD_SHEDDING_COST)

    summary = {
        "objective_total_cost": objective,
        "generation_cost": generation_cost,
        "load_shedding_cost": load_shedding_cost,
        "total_generation_mwh": float(dispatch.sum()),
        "total_demand_mwh": float(demand_matrix.sum()),
        "total_load_shed_mwh": float(shed.sum()),
        "total_wind_available_mwh": float(wind_available.to_numpy().sum()) if wind_idx else 0.0,
        "total_wind_generation_mwh": float(wind_generation.to_numpy().sum()) if wind_idx else 0.0,
        "total_wind_curtailment_mwh": float(wind_curtailment.to_numpy().sum()) if wind_idx else 0.0,
        "peak_system_demand_mw": float(demand_matrix.sum(axis=1).max()),
        "peak_gas_generation_mw": float(dispatch_by_carrier.get("gas", pd.Series(np.zeros(hours))).max()),
        "peak_wind_generation_mw": float(dispatch_by_carrier.get("onshore wind", pd.Series(np.zeros(hours))).max()),
        "peak_nuclear_generation_mw": float(dispatch_by_carrier.get("nuclear", pd.Series(np.zeros(hours))).max()),
        "max_hourly_load_shed_mw": float(shed_df.sum(axis=1).max()),
        "solver_status": int(result.status),
        "solver_message": result.message,
    }

    return {
        "summary": summary,
        "buses": buses,
        "links": links,
        "generators": generators,
        "storage": storage,
        "demand": demand,
        "wind_cf": wind_cf,
        "dispatch": dispatch,
        "dispatch_by_carrier": dispatch_by_carrier,
        "generator_dispatch": generator_dispatch,
        "wind_available": wind_available,
        "wind_generation": wind_generation,
        "wind_curtailment": wind_curtailment,
        "storage_charge": storage_charge_df,
        "storage_discharge": storage_discharge_df,
        "storage_soc": storage_soc_df,
        "flows": flow_df,
        "load_shedding": shed_df,
        "hourly_system_demand": pd.Series(demand_matrix.sum(axis=1), name="system_demand_mw"),
        "hourly_wind_cf_mean": wind_cf[bus_order].mean(axis=1),
    }


def save_outputs(data: dict[str, object], validation_summary: dict[str, object]) -> None:
    buses = data["buses"]
    links = data["links"]
    generators = data["generators"]
    storage = data["storage"]
    demand = data["demand"]
    dispatch = data["dispatch"]
    dispatch_by_carrier = data["dispatch_by_carrier"]
    wind_available = data["wind_available"]
    wind_generation = data["wind_generation"]
    wind_curtailment = data["wind_curtailment"]
    storage_charge = data["storage_charge"]
    storage_discharge = data["storage_discharge"]
    storage_soc = data["storage_soc"]
    flows = data["flows"]
    load_shedding = data["load_shedding"]

    with open(OUTPUT_DIR / "data_validation_summary.json", "w", encoding="utf-8") as f:
        json.dump(validation_summary, f, indent=2)

    with open(OUTPUT_DIR / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(data["summary"], f, indent=2)

    pd.DataFrame({
        "hour": np.arange(len(demand)),
        "system_demand_mw": demand.sum(axis=1).to_numpy(),
        "mean_wind_cf": data["hourly_wind_cf_mean"].to_numpy(),
        "load_shedding_mw": load_shedding.sum(axis=1).to_numpy(),
    }).to_csv(OUTPUT_DIR / "hourly_system_summary.csv", index=False)

    dispatch_carrier_out = dispatch_by_carrier.copy()
    dispatch_carrier_out.index = np.arange(len(dispatch_carrier_out))
    dispatch_carrier_out.to_csv(OUTPUT_DIR / "dispatch_by_carrier_hourly.csv", index_label="hour")

    gen_dispatch_long = []
    for g, row in generators.iterrows():
        tmp = pd.DataFrame({
            "hour": np.arange(dispatch.shape[0]),
            "generator_id": g,
            "bus": row["bus"],
            "carrier": row["carrier"],
            "dispatch_mw": dispatch[:, g],
            "available_mw": wind_available[g].to_numpy() if row["carrier"] == "onshore wind" else float(row["p_nom"]),
        })
        gen_dispatch_long.append(tmp)
    pd.concat(gen_dispatch_long, ignore_index=True).to_csv(OUTPUT_DIR / "generator_dispatch_hourly.csv", index=False)

    if not wind_generation.empty:
        wind_bus_map = generators.loc[generators["carrier"] == "onshore wind", "bus"].reset_index(drop=True)
        wind_gen_by_bus = wind_generation.copy()
        wind_gen_by_bus.columns = wind_bus_map
        wind_gen_by_bus = wind_gen_by_bus.groupby(level=0, axis=1).sum()
        wind_av_by_bus = wind_available.copy()
        wind_av_by_bus.columns = wind_bus_map
        wind_av_by_bus = wind_av_by_bus.groupby(level=0, axis=1).sum()
        wind_curt_by_bus = wind_curtailment.copy()
        wind_curt_by_bus.columns = wind_bus_map
        wind_curt_by_bus = wind_curt_by_bus.groupby(level=0, axis=1).sum()
        wind_gen_by_bus.to_csv(OUTPUT_DIR / "wind_generation_by_bus_hourly.csv", index_label="hour")
        wind_av_by_bus.to_csv(OUTPUT_DIR / "wind_available_by_bus_hourly.csv", index_label="hour")
        wind_curt_by_bus.to_csv(OUTPUT_DIR / "wind_curtailment_by_bus_hourly.csv", index_label="hour")

    if len(storage) > 0:
        charge_out = storage_charge.copy()
        discharge_out = storage_discharge.copy()
        soc_out = storage_soc.copy()
        charge_out.columns = [f"{storage.loc[s, 'bus']}_{storage.loc[s, 'carrier']}_charge_mw" for s in storage.index]
        discharge_out.columns = [f"{storage.loc[s, 'bus']}_{storage.loc[s, 'carrier']}_discharge_mw" for s in storage.index]
        soc_out.columns = [f"{storage.loc[s, 'bus']}_{storage.loc[s, 'carrier']}_soc_mwh" for s in storage.index]
        pd.concat([charge_out, discharge_out, soc_out], axis=1).to_csv(OUTPUT_DIR / "storage_operations_hourly.csv", index_label="hour")

    flow_out = flows.copy()
    flow_out.columns = [f"{links.loc[l, 'bus0']}__{links.loc[l, 'bus1']}" for l in links.index]
    flow_out.to_csv(OUTPUT_DIR / "line_flows_hourly.csv", index_label="hour")

    line_loading = pd.DataFrame({
        "link_id": links.index,
        "bus0": links["bus0"],
        "bus1": links["bus1"],
        "capacity_mw": links["p_nom"],
        "max_abs_flow_mw": np.abs(flows.to_numpy()).max(axis=0),
        "mean_abs_flow_mw": np.abs(flows.to_numpy()).mean(axis=0),
    })
    line_loading["max_loading_pct"] = 100.0 * line_loading["max_abs_flow_mw"] / line_loading["capacity_mw"]
    line_loading["mean_loading_pct"] = 100.0 * line_loading["mean_abs_flow_mw"] / line_loading["capacity_mw"]
    line_loading.sort_values("max_loading_pct", ascending=False).to_csv(OUTPUT_DIR / "line_loading_summary.csv", index=False)

    load_shedding.to_csv(OUTPUT_DIR / "load_shedding_by_bus_hourly.csv", index_label="hour")

    carrier_energy = dispatch_by_carrier.sum(axis=0).rename("energy_mwh").reset_index().rename(columns={"carrier": "carrier"})
    carrier_cost = generators.assign(
        dispatched_mwh=dispatch.sum(axis=0),
        variable_cost=lambda df: df["dispatched_mwh"] * df["marginal_cost"],
    ).groupby("carrier")[["dispatched_mwh", "variable_cost"]].sum().reset_index()
    carrier_summary = carrier_energy.merge(carrier_cost, on="carrier", how="outer")
    carrier_summary.to_csv(OUTPUT_DIR / "carrier_summary.csv", index=False)

    bus_summary = pd.DataFrame({
        "bus": buses["name"],
        "total_demand_mwh": demand[buses["name"]].sum().to_numpy(),
    })
    generation_by_bus = generators.assign(total_dispatch_mwh=dispatch.sum(axis=0)).groupby("bus")["total_dispatch_mwh"].sum()
    bus_summary["total_generation_mwh"] = bus_summary["bus"].map(generation_by_bus).fillna(0.0)
    bus_summary["total_load_shed_mwh"] = bus_summary["bus"].map(load_shedding.sum(axis=0)).fillna(0.0)
    bus_summary.to_csv(OUTPUT_DIR / "bus_summary.csv", index=False)


def create_figures(data: dict[str, object]) -> None:
    buses = data["buses"]
    links = data["links"]
    demand = data["demand"]
    generators = data["generators"]
    storage = data["storage"]
    dispatch_by_carrier = data["dispatch_by_carrier"]
    load_shedding = data["load_shedding"]
    flows = data["flows"]
    storage_soc = data["storage_soc"]
    storage_charge = data["storage_charge"]
    storage_discharge = data["storage_discharge"]
    hourly_system_demand = data["hourly_system_demand"]
    hourly_wind_cf_mean = data["hourly_wind_cf_mean"]
    wind_curtailment = data["wind_curtailment"]

    # Figure 1: network overview
    fig, ax = plt.subplots(figsize=(10, 8))
    for _, row in links.iterrows():
        b0 = buses.loc[buses["name"] == row["bus0"]].iloc[0]
        b1 = buses.loc[buses["name"] == row["bus1"]].iloc[0]
        ax.plot([b0["x"], b1["x"]], [b0["y"], b1["y"]], color="lightgray", linewidth=0.8 + row["p_nom"] / 4000.0, zorder=1)
    demand_total = demand[buses["name"]].sum(axis=0)
    ax.scatter(buses["x"], buses["y"], s=40 + demand_total / 300.0, c=demand_total, cmap="viridis", edgecolor="black", zorder=2)
    for _, row in buses.iterrows():
        ax.text(row["x"] + 0.08, row["y"] + 0.05, row["name"], fontsize=8)
    ax.set_title("GB test network overview: buses sized by weekly demand")
    ax.set_xlabel("Longitude-like coordinate")
    ax.set_ylabel("Latitude-like coordinate")
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=float(demand_total.min()), vmax=float(demand_total.max())))
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Weekly bus demand (MWh)")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "network_overview.png", dpi=200)
    plt.close(fig)

    # Figure 2: demand and wind conditions
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(hourly_system_demand.index, hourly_system_demand.values, color="tab:blue", label="System demand")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Demand (MW)", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax2 = ax1.twinx()
    ax2.plot(hourly_wind_cf_mean.index, hourly_wind_cf_mean.values, color="tab:green", label="Mean wind CF")
    ax2.set_ylabel("Mean wind capacity factor", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:green")
    ax1.set_title("System demand and average wind availability over the week")
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "demand_wind_profiles.png", dpi=200)
    plt.close(fig)

    # Figure 3: dispatch stack
    fig, ax = plt.subplots(figsize=(12, 6))
    carriers = [c for c in ["onshore wind", "nuclear", "gas"] if c in dispatch_by_carrier.columns]
    stack = [dispatch_by_carrier[c].to_numpy() for c in carriers]
    ax.stackplot(np.arange(len(dispatch_by_carrier)), stack, labels=carriers, alpha=0.9)
    ax.plot(load_shedding.sum(axis=1).to_numpy(), color="red", linewidth=1.5, label="Load shedding")
    ax.plot(hourly_system_demand.to_numpy(), color="black", linewidth=1.2, linestyle="--", label="Demand")
    ax.set_title("Optimal hourly dispatch by carrier")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (MW)")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "dispatch_stack.png", dpi=200)
    plt.close(fig)

    # Figure 4: storage operation
    if len(storage) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        for s, row in storage.iterrows():
            label = f"{row['bus']} {row['carrier']}"
            axes[0].plot(storage_soc.index, storage_soc[s], label=label)
            axes[1].plot(storage_charge.index, storage_charge[s], linestyle="--", label=f"{label} charge")
            axes[1].plot(storage_discharge.index, storage_discharge[s], linestyle="-", label=f"{label} discharge")
        axes[0].set_ylabel("State of charge (MWh)")
        axes[0].set_title("Storage state of charge")
        axes[0].legend(loc="upper right", fontsize=8)
        axes[1].set_ylabel("Power (MW)")
        axes[1].set_xlabel("Hour")
        axes[1].set_title("Storage charging and discharging")
        axes[1].legend(loc="upper right", ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(IMAGE_DIR / "storage_operation.png", dpi=200)
        plt.close(fig)

    # Figure 5: transmission loading
    fig, ax = plt.subplots(figsize=(12, 5))
    loading_pct = 100.0 * np.abs(flows.to_numpy()) / links["p_nom"].to_numpy()[None, :]
    top_links = np.argsort(loading_pct.max(axis=0))[-5:][::-1]
    for idx in top_links:
        label = f"{links.loc[idx, 'bus0']}→{links.loc[idx, 'bus1']}"
        ax.plot(loading_pct[:, idx], label=label)
    ax.set_title("Hourly loading of the five most-constrained transmission links")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Line loading (% of capacity)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "transmission_loading.png", dpi=200)
    plt.close(fig)

    # Figure 6: curtailment and unmet demand diagnostic
    fig, ax = plt.subplots(figsize=(12, 5))
    curtailment_series = wind_curtailment.sum(axis=1) if not wind_curtailment.empty else pd.Series(np.zeros(len(hourly_system_demand)))
    ax.plot(curtailment_series.index, curtailment_series.values, label="Wind curtailment", color="tab:orange")
    ax.plot(load_shedding.sum(axis=1).index, load_shedding.sum(axis=1).values, label="Load shedding", color="tab:red")
    ax.set_title("Renewable curtailment and unserved load")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (MW)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(IMAGE_DIR / "curtailment_and_unserved_energy.png", dpi=200)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    data = load_data()
    validation_summary = validate_data(data)
    results = build_and_solve_model(data)
    save_outputs(results, validation_summary)
    create_figures(results)
    print("Analysis complete. Outputs written to outputs/ and report/images/.")


if __name__ == "__main__":
    main()
