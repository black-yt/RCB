#!/usr/bin/env python3
"""Run a reproducible GB dispatch study on the provided open dataset."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.optimize import linprog
from scipy.sparse import lil_matrix


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
REPORT_IMG_DIR = ROOT / "report" / "images"

VOLL_GBP_PER_MWH = 6000.0


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    generator_scale: dict[str, float]
    storage_scale: float = 1.0
    link_scale: float = 1.0
    gas_cost_multiplier: float = 1.0


SCENARIOS = [
    Scenario(
        name="stress_base",
        description="Raw provided capacities with network constraints and storage.",
        generator_scale={"onshore wind": 1.0, "gas": 1.0, "nuclear": 1.0},
        storage_scale=1.0,
        link_scale=1.0,
    ),
    Scenario(
        name="adequacy_scaled",
        description="All generation and storage scaled by 4.5x to approximate an adequate future system.",
        generator_scale={"onshore wind": 4.5, "gas": 4.5, "nuclear": 4.5},
        storage_scale=4.5,
        link_scale=1.0,
    ),
    Scenario(
        name="adequacy_copperplate",
        description="Adequacy-scaled system with effectively unconstrained transmission.",
        generator_scale={"onshore wind": 4.5, "gas": 4.5, "nuclear": 4.5},
        storage_scale=4.5,
        link_scale=100.0,
    ),
    Scenario(
        name="adequacy_no_storage",
        description="Adequacy-scaled system with storage removed to quantify flexibility value.",
        generator_scale={"onshore wind": 4.5, "gas": 4.5, "nuclear": 4.5},
        storage_scale=0.0,
        link_scale=1.0,
    ),
    Scenario(
        name="adequacy_high_gas_price",
        description="Adequacy-scaled system with doubled gas marginal costs.",
        generator_scale={"onshore wind": 4.5, "gas": 4.5, "nuclear": 4.5},
        storage_scale=4.5,
        link_scale=1.0,
        gas_cost_multiplier=2.0,
    ),
]


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "buses": pd.read_csv(DATA_DIR / "buses.csv"),
        "links": pd.read_csv(DATA_DIR / "links.csv"),
        "demand": pd.read_csv(DATA_DIR / "demand.csv"),
        "generators": pd.read_csv(DATA_DIR / "generators.csv"),
        "wind_cf": pd.read_csv(DATA_DIR / "wind_cf.csv"),
        "storage": pd.read_csv(DATA_DIR / "storage.csv"),
    }


def prepare_base_objects(data: dict[str, pd.DataFrame]) -> dict[str, object]:
    buses = data["buses"].copy()
    links = data["links"].copy()
    generators = data["generators"].copy()
    storage = data["storage"].copy()
    demand = data["demand"].copy()
    wind_cf = data["wind_cf"].copy()

    bus_names = buses["name"].tolist()
    bus_to_idx = {bus: idx for idx, bus in enumerate(bus_names)}
    n_buses = len(bus_names)
    n_links = len(links)
    n_gens = len(generators)
    n_storage = len(storage)
    n_hours = len(demand)

    gen_bus_idx = generators["bus"].map(bus_to_idx).to_numpy()
    stor_bus_idx = storage["bus"].map(bus_to_idx).to_numpy()

    incidence = np.zeros((n_buses, n_links))
    for line_idx, row in links.iterrows():
        incidence[bus_to_idx[row["bus0"]], line_idx] = -1.0
        incidence[bus_to_idx[row["bus1"]], line_idx] = 1.0

    wind_availability = np.ones((n_hours, n_gens))
    for gen_idx, row in generators.iterrows():
        if row["carrier"] == "onshore wind":
            wind_availability[:, gen_idx] = wind_cf[row["bus"]].to_numpy()

    return {
        "buses": buses,
        "links": links,
        "generators": generators,
        "storage": storage,
        "demand": demand,
        "wind_cf": wind_cf,
        "bus_names": bus_names,
        "bus_to_idx": bus_to_idx,
        "gen_bus_idx": gen_bus_idx,
        "stor_bus_idx": stor_bus_idx,
        "incidence": incidence,
        "wind_availability": wind_availability,
        "n_buses": n_buses,
        "n_links": n_links,
        "n_gens": n_gens,
        "n_storage": n_storage,
        "n_hours": n_hours,
    }


def build_scenario_inputs(base: dict[str, object], scenario: Scenario) -> dict[str, object]:
    generators = base["generators"].copy()
    storage = base["storage"].copy()
    links = base["links"].copy()

    generators["scale"] = generators["carrier"].map(scenario.generator_scale).fillna(1.0)
    generators["p_nom_scaled"] = generators["p_nom"] * generators["scale"]
    generators["marginal_cost_scaled"] = generators["marginal_cost"]
    gas_mask = generators["carrier"] == "gas"
    generators.loc[gas_mask, "marginal_cost_scaled"] *= scenario.gas_cost_multiplier

    storage["p_nom_scaled"] = storage["p_nom"] * scenario.storage_scale
    storage["e_nom_scaled"] = storage["e_nom"] * scenario.storage_scale
    storage = storage.loc[
        (storage["p_nom_scaled"] > 1e-9) & (storage["e_nom_scaled"] > 1e-9)
    ].reset_index(drop=True)
    links["p_nom_scaled"] = links["p_nom"] * scenario.link_scale

    return {"generators": generators, "storage": storage, "links": links}


def solve_dispatch(base: dict[str, object], scenario: Scenario) -> dict[str, object]:
    scenario_inputs = build_scenario_inputs(base, scenario)
    generators = scenario_inputs["generators"]
    storage = scenario_inputs["storage"]
    links = scenario_inputs["links"]

    demand = base["demand"].to_numpy()
    wind_availability = base["wind_availability"]
    gen_bus_idx = base["gen_bus_idx"]
    incidence = base["incidence"]

    n_hours = base["n_hours"]
    n_buses = base["n_buses"]
    n_links = base["n_links"]
    n_gens = base["n_gens"]
    n_storage = len(storage)
    stor_bus_idx = storage["bus"].map(base["bus_to_idx"]).to_numpy() if n_storage else np.array([], dtype=int)

    n_gen_vars = n_hours * n_gens
    n_flow_vars = n_hours * n_links
    n_charge_vars = n_hours * n_storage
    n_discharge_vars = n_hours * n_storage
    n_energy_vars = n_hours * n_storage
    n_shed_vars = n_hours * n_buses
    total_vars = (
        n_gen_vars
        + n_flow_vars
        + n_charge_vars
        + n_discharge_vars
        + n_energy_vars
        + n_shed_vars
    )

    offsets = {}
    cursor = 0
    for key, size in [
        ("gen", n_gen_vars),
        ("flow", n_flow_vars),
        ("charge", n_charge_vars),
        ("discharge", n_discharge_vars),
        ("energy", n_energy_vars),
        ("shed", n_shed_vars),
    ]:
        offsets[key] = cursor
        cursor += size

    def idx(block: str, t: int, i: int) -> int:
        width = {
            "gen": n_gens,
            "flow": n_links,
            "charge": n_storage,
            "discharge": n_storage,
            "energy": n_storage,
            "shed": n_buses,
        }[block]
        return offsets[block] + t * width + i

    c = np.zeros(total_vars)
    bounds: list[tuple[float | None, float | None]] = [(0.0, 0.0)] * total_vars

    avail = wind_availability * generators["p_nom_scaled"].to_numpy()
    mc = generators["marginal_cost_scaled"].to_numpy()
    link_caps = links["p_nom_scaled"].to_numpy()
    storage_p = storage["p_nom_scaled"].to_numpy()
    storage_e = storage["e_nom_scaled"].to_numpy()
    storage_eff = storage["efficiency"].to_numpy()

    for t in range(n_hours):
        for g in range(n_gens):
            c[idx("gen", t, g)] = mc[g]
            bounds[idx("gen", t, g)] = (0.0, float(avail[t, g]))
        for l in range(n_links):
            cap = float(link_caps[l])
            bounds[idx("flow", t, l)] = (-cap, cap)
        for s in range(n_storage):
            p_cap = float(storage_p[s])
            e_cap = float(storage_e[s])
            bounds[idx("charge", t, s)] = (0.0, p_cap)
            bounds[idx("discharge", t, s)] = (0.0, p_cap)
            bounds[idx("energy", t, s)] = (0.0, e_cap)
        for b in range(n_buses):
            c[idx("shed", t, b)] = VOLL_GBP_PER_MWH
            bounds[idx("shed", t, b)] = (0.0, float(demand[t, b]))

    n_balance_eq = n_hours * n_buses
    n_storage_eq = n_hours * n_storage
    A_eq = lil_matrix((n_balance_eq + n_storage_eq, total_vars))
    b_eq = np.zeros(n_balance_eq + n_storage_eq)

    row = 0
    for t in range(n_hours):
        for b in range(n_buses):
            b_eq[row] = demand[t, b]
            gen_ids = np.where(gen_bus_idx == b)[0]
            for g in gen_ids:
                A_eq[row, idx("gen", t, g)] = 1.0
            flow_ids = np.where(incidence[b] != 0.0)[0]
            for l in flow_ids:
                A_eq[row, idx("flow", t, l)] = incidence[b, l]
            stor_ids = np.where(stor_bus_idx == b)[0]
            for s in stor_ids:
                A_eq[row, idx("charge", t, s)] = -1.0
                A_eq[row, idx("discharge", t, s)] = 1.0
            A_eq[row, idx("shed", t, b)] = 1.0
            row += 1

    for t in range(n_hours):
        prev_t = (t - 1) % n_hours
        for s in range(n_storage):
            eff = float(storage_eff[s])
            A_eq[row, idx("energy", t, s)] = 1.0
            A_eq[row, idx("energy", prev_t, s)] = -1.0
            A_eq[row, idx("charge", t, s)] = -eff
            A_eq[row, idx("discharge", t, s)] = 1.0 / eff
            b_eq[row] = 0.0
            row += 1

    result = linprog(
        c=c,
        A_eq=A_eq.tocsr(),
        b_eq=b_eq,
        bounds=bounds,
        method="highs-ipm",
        options={"disp": False},
    )
    if not result.success:
        raise RuntimeError(f"Optimization failed for {scenario.name}: {result.message}")

    x = result.x
    gen = x[offsets["gen"] : offsets["gen"] + n_gen_vars].reshape(n_hours, n_gens)
    flow = x[offsets["flow"] : offsets["flow"] + n_flow_vars].reshape(n_hours, n_links)
    charge = x[offsets["charge"] : offsets["charge"] + n_charge_vars].reshape(n_hours, n_storage)
    discharge = x[offsets["discharge"] : offsets["discharge"] + n_discharge_vars].reshape(n_hours, n_storage)
    energy = x[offsets["energy"] : offsets["energy"] + n_energy_vars].reshape(n_hours, n_storage)
    shed = x[offsets["shed"] : offsets["shed"] + n_shed_vars].reshape(n_hours, n_buses)

    wind_mask = generators["carrier"] == "onshore wind"
    wind_available = pd.DataFrame(
        avail[:, wind_mask.to_numpy()],
        columns=generators.loc[wind_mask, "bus"].tolist(),
    )
    wind_dispatched = pd.DataFrame(
        gen[:, wind_mask.to_numpy()],
        columns=generators.loc[wind_mask, "bus"].tolist(),
    )
    wind_curtailment = wind_available - wind_dispatched

    hourly_dispatch = pd.DataFrame(gen, columns=[f"gen_{i}" for i in range(n_gens)])
    hourly_dispatch_by_carrier = pd.DataFrame(index=np.arange(n_hours))
    for carrier in generators["carrier"].unique():
        carrier_mask = generators["carrier"] == carrier
        hourly_dispatch_by_carrier[carrier] = gen[:, carrier_mask.to_numpy()].sum(axis=1)
    hourly_dispatch_by_carrier["charge"] = charge.sum(axis=1)
    hourly_dispatch_by_carrier["discharge"] = discharge.sum(axis=1)
    hourly_dispatch_by_carrier["load_shedding"] = shed.sum(axis=1)
    hourly_dispatch_by_carrier["demand"] = demand.sum(axis=1)
    hourly_dispatch_by_carrier["wind_available"] = wind_available.sum(axis=1)
    hourly_dispatch_by_carrier["wind_curtailment"] = wind_curtailment.sum(axis=1)
    system_balance_residual = (
        hourly_dispatch_by_carrier["onshore wind"]
        + hourly_dispatch_by_carrier.get("nuclear", 0.0)
        + hourly_dispatch_by_carrier.get("gas", 0.0)
        + hourly_dispatch_by_carrier["discharge"]
        - hourly_dispatch_by_carrier["charge"]
        + hourly_dispatch_by_carrier["load_shedding"]
        - hourly_dispatch_by_carrier["demand"]
    )

    line_loading = np.abs(flow) / np.maximum(link_caps.reshape(1, -1), 1e-9)

    total_demand = float(demand.sum())
    total_shed = float(shed.sum())
    served_demand = total_demand - total_shed
    wind_dispatch_total = float(wind_dispatched.sum().sum())
    wind_available_total = float(wind_available.sum().sum())
    gas_dispatch_total = float(gen[:, (generators["carrier"] == "gas").to_numpy()].sum())
    nuclear_dispatch_total = float(gen[:, (generators["carrier"] == "nuclear").to_numpy()].sum())
    curtailment_total = float(wind_curtailment.sum().sum())
    storage_discharge_total = float(discharge.sum())
    storage_charge_total = float(charge.sum())
    congestion_hours = int((line_loading >= 0.95).any(axis=1).sum())

    summary = {
        "scenario": scenario.name,
        "description": scenario.description,
        "objective_gbp": float(result.fun),
        "total_demand_mwh": total_demand,
        "served_demand_mwh": served_demand,
        "load_shedding_mwh": total_shed,
        "load_shedding_pct": 100.0 * total_shed / total_demand,
        "wind_available_mwh": wind_available_total,
        "wind_dispatch_mwh": wind_dispatch_total,
        "wind_curtailment_mwh": curtailment_total,
        "wind_curtailment_pct": 100.0 * curtailment_total / max(wind_available_total, 1e-9),
        "gas_dispatch_mwh": gas_dispatch_total,
        "nuclear_dispatch_mwh": nuclear_dispatch_total,
        "storage_charge_mwh": storage_charge_total,
        "storage_discharge_mwh": storage_discharge_total,
        "renewable_share_pct": 100.0 * wind_dispatch_total / max(served_demand, 1e-9),
        "peak_shed_mw": float(shed.sum(axis=1).max()),
        "congestion_hours": congestion_hours,
        "max_line_loading_pct": float(100.0 * line_loading.max()),
        "avg_line_loading_pct": float(100.0 * line_loading.mean()),
        "max_abs_system_balance_residual_mw": float(np.abs(system_balance_residual).max()),
    }

    return {
        "scenario": scenario,
        "generators": generators,
        "storage": storage,
        "links": links,
        "summary": summary,
        "hourly_dispatch_by_carrier": hourly_dispatch_by_carrier,
        "hourly_generation_raw": pd.DataFrame(gen, columns=generators.index),
        "flows": pd.DataFrame(flow, columns=links.index),
        "charge": pd.DataFrame(charge, columns=storage["bus"].tolist()),
        "discharge": pd.DataFrame(discharge, columns=storage["bus"].tolist()),
        "energy": pd.DataFrame(energy, columns=storage["bus"].tolist()),
        "shed": pd.DataFrame(shed, columns=base["bus_names"]),
        "wind_available": wind_available,
        "wind_dispatch": wind_dispatched,
        "wind_curtailment": wind_curtailment,
        "line_loading": pd.DataFrame(line_loading, columns=links.index),
    }


def save_results(base: dict[str, object], scenario_result: dict[str, object]) -> None:
    name = scenario_result["scenario"].name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    scenario_result["hourly_dispatch_by_carrier"].to_csv(
        OUTPUT_DIR / f"{name}_dispatch_by_carrier_hourly.csv", index_label="hour"
    )
    scenario_result["flows"].to_csv(
        OUTPUT_DIR / f"{name}_line_flows_hourly.csv", index_label="hour"
    )
    scenario_result["line_loading"].to_csv(
        OUTPUT_DIR / f"{name}_line_loading_hourly.csv", index_label="hour"
    )
    scenario_result["shed"].to_csv(
        OUTPUT_DIR / f"{name}_load_shedding_by_bus_hourly.csv", index_label="hour"
    )
    scenario_result["energy"].to_csv(
        OUTPUT_DIR / f"{name}_storage_energy_hourly.csv", index_label="hour"
    )
    scenario_result["charge"].to_csv(
        OUTPUT_DIR / f"{name}_storage_charge_hourly.csv", index_label="hour"
    )
    scenario_result["discharge"].to_csv(
        OUTPUT_DIR / f"{name}_storage_discharge_hourly.csv", index_label="hour"
    )
    scenario_result["wind_available"].to_csv(
        OUTPUT_DIR / f"{name}_wind_available_hourly.csv", index_label="hour"
    )
    scenario_result["wind_dispatch"].to_csv(
        OUTPUT_DIR / f"{name}_wind_dispatch_hourly.csv", index_label="hour"
    )
    scenario_result["wind_curtailment"].to_csv(
        OUTPUT_DIR / f"{name}_wind_curtailment_hourly.csv", index_label="hour"
    )
    pd.DataFrame([scenario_result["summary"]]).to_csv(
        OUTPUT_DIR / f"{name}_summary.csv", index=False
    )


def plot_network_overview(base: dict[str, object], scenario_results: list[dict[str, object]]) -> None:
    buses = base["buses"]
    links = base["links"]
    demand = base["demand"]
    generators = base["generators"]
    storage = base["storage"]

    demand_total = demand.sum()
    gen_cap = generators.groupby("bus")["p_nom"].sum()
    storage_cap = storage.groupby("bus")["p_nom"].sum()

    fig, ax = plt.subplots(figsize=(10, 7))
    for _, row in links.iterrows():
        p0 = buses.loc[buses["name"] == row["bus0"], ["x", "y"]].iloc[0]
        p1 = buses.loc[buses["name"] == row["bus1"], ["x", "y"]].iloc[0]
        ax.plot([p0["x"], p1["x"]], [p0["y"], p1["y"]], color="#9aa0a6", linewidth=1.2, zorder=1)

    sc = ax.scatter(
        buses["x"],
        buses["y"],
        s=demand_total.reindex(buses["name"]).to_numpy() / 1200.0,
        c=gen_cap.reindex(buses["name"]).fillna(0.0),
        cmap="YlGnBu",
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
    )
    for _, row in buses.iterrows():
        ax.text(row["x"] + 0.08, row["y"] + 0.05, row["name"], fontsize=8)
    for _, row in storage.iterrows():
        bus = buses.loc[buses["name"] == row["bus"], ["x", "y"]].iloc[0]
        ax.scatter(
            [bus["x"]],
            [bus["y"]],
            marker="s",
            s=90,
            color="#d55e00",
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )

    ax.set_title("GB Network Overview")
    ax.set_xlabel("Longitude-like coordinate")
    ax.set_ylabel("Latitude-like coordinate")
    cbar = fig.colorbar(sc, ax=ax, label="Installed generation capacity (MW)")
    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray", markeredgecolor="black",
               markersize=10, label="Bus size ~ weekly demand"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#d55e00", markeredgecolor="black",
               markersize=8, label="Storage bus"),
    ]
    ax.legend(handles=legend_items, loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "network_overview.png", dpi=200)
    plt.close(fig)


def plot_demand_wind_overview(base: dict[str, object]) -> None:
    demand = base["demand"].sum(axis=1)
    wind_cf = base["wind_cf"].mean(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, height_ratios=[2, 1.3])

    axes[0].plot(demand.index, demand.to_numpy(), color="#1f77b4", linewidth=2)
    axes[0].set_ylabel("System demand (MW)")
    axes[0].set_title("Hourly System Demand and Mean Wind Capacity Factor")
    axes[0].grid(alpha=0.25)

    axes[1].plot(wind_cf.index, wind_cf.to_numpy(), color="#2ca02c", linewidth=2)
    axes[1].fill_between(wind_cf.index, 0, wind_cf.to_numpy(), color="#2ca02c", alpha=0.2)
    axes[1].set_ylabel("Mean wind CF")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "demand_wind_overview.png", dpi=200)
    plt.close(fig)


def plot_dispatch_timeseries(results_map: dict[str, dict[str, object]]) -> None:
    dispatch = results_map["adequacy_scaled"]["hourly_dispatch_by_carrier"].copy()
    fig, ax = plt.subplots(figsize=(13, 6))

    stack_cols = ["onshore wind", "nuclear", "gas", "discharge", "load_shedding"]
    colors = ["#4daf4a", "#984ea3", "#ff7f00", "#377eb8", "#e41a1c"]
    ax.stackplot(
        dispatch.index,
        [dispatch[c] for c in stack_cols],
        labels=stack_cols,
        colors=colors,
        alpha=0.85,
    )
    ax.plot(dispatch.index, dispatch["demand"], color="black", linewidth=1.8, label="Demand")
    ax.plot(
        dispatch.index,
        dispatch["wind_available"],
        color="#1b9e77",
        linewidth=1.4,
        linestyle="--",
        label="Available wind",
    )
    ax.set_title("Adequacy-Scaled Scenario: Hourly Dispatch")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Power (MW)")
    ax.legend(ncol=3, fontsize=9, frameon=True)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "adequacy_dispatch.png", dpi=200)
    plt.close(fig)


def plot_scenario_comparison(summary_df: pd.DataFrame) -> None:
    order = [
        "stress_base",
        "adequacy_scaled",
        "adequacy_copperplate",
        "adequacy_no_storage",
        "adequacy_high_gas_price",
    ]
    plot_df = summary_df.set_index("scenario").loc[order].reset_index()
    labels = [
        "Stress\nBase",
        "Adequacy\nScaled",
        "Copperplate",
        "No\nStorage",
        "High\nGas Price",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].bar(labels, plot_df["objective_gbp"] / 1e9, color="#4c78a8")
    axes[0].set_ylabel("Total system cost (billion GBP)")
    axes[0].set_title("Scenario Costs")
    axes[0].grid(axis="y", alpha=0.25)

    x = np.arange(len(labels))
    width = 0.38
    axes[1].bar(x - width / 2, plot_df["load_shedding_pct"], width, label="Load shedding (%)", color="#e45756")
    axes[1].bar(x + width / 2, plot_df["wind_curtailment_pct"], width, label="Wind curtailment (%)", color="#72b7b2")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Percent")
    axes[1].set_title("Reliability and Renewable Integration")
    axes[1].legend(frameon=True)
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "scenario_comparison.png", dpi=200)
    plt.close(fig)


def plot_congestion_and_storage(results_map: dict[str, dict[str, object]], base: dict[str, object]) -> None:
    adequacy = results_map["adequacy_scaled"]
    line_loading = adequacy["line_loading"].copy()
    links = base["links"].copy()
    line_names = links["bus0"] + "-" + links["bus1"]
    top_lines = line_loading.mean().sort_values(ascending=False).head(10).index.tolist()
    top_labels = line_names.iloc[top_lines].tolist()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    axes[0].barh(
        top_labels[::-1],
        (100 * line_loading[top_lines].mean()).to_numpy()[::-1],
        color="#f28e2b",
    )
    axes[0].set_xlabel("Average loading (%)")
    axes[0].set_title("Most Utilized Transmission Lines\n(Adequacy-Scaled)")
    axes[0].grid(axis="x", alpha=0.25)

    soc = adequacy["energy"].copy()
    for col in soc.columns:
        axes[1].plot(soc.index, soc[col], linewidth=1.8, label=col)
    axes[1].set_title("Storage State of Charge")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Energy (MWh)")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=True)

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "congestion_storage.png", dpi=200)
    plt.close(fig)


def plot_validation(summary_df: pd.DataFrame) -> None:
    plot_df = summary_df.copy()
    labels = {
        "stress_base": "Stress Base",
        "adequacy_scaled": "Adequacy Scaled",
        "adequacy_copperplate": "Copperplate",
        "adequacy_no_storage": "No Storage",
        "adequacy_high_gas_price": "High Gas Price",
    }
    plot_df["label"] = plot_df["scenario"].map(labels)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(
        plot_df["label"],
        np.maximum(plot_df["max_abs_system_balance_residual_mw"], 1e-9),
        color="#59a14f",
    )
    ax.set_yscale("log")
    ax.set_ylabel("Max hourly system balance residual (MW, log scale)")
    ax.set_title("Model Validation: Power Balance Residuals")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / "validation_balance.png", dpi=200)
    plt.close(fig)


def main() -> None:
    sns.set_theme(style="whitegrid")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_IMG_DIR.mkdir(parents=True, exist_ok=True)

    data = load_inputs()
    base = prepare_base_objects(data)

    scenario_results = []
    for scenario in SCENARIOS:
        print(f"Solving {scenario.name}...", flush=True)
        scenario_result = solve_dispatch(base, scenario)
        save_results(base, scenario_result)
        scenario_results.append(scenario_result)
        print(
            f"Finished {scenario.name}: cost={scenario_result['summary']['objective_gbp']:.2f} GBP, "
            f"shed={scenario_result['summary']['load_shedding_pct']:.2f}%",
            flush=True,
        )

    summary_df = pd.DataFrame([result["summary"] for result in scenario_results])
    summary_df.to_csv(OUTPUT_DIR / "scenario_summary.csv", index=False)

    results_map = {result["scenario"].name: result for result in scenario_results}
    plot_network_overview(base, scenario_results)
    plot_demand_wind_overview(base)
    plot_dispatch_timeseries(results_map)
    plot_scenario_comparison(summary_df)
    plot_congestion_and_storage(results_map, base)
    plot_validation(summary_df)

    print("Completed analysis.")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
