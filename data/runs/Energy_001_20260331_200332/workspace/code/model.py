import pypsa
import pandas as pd
import numpy as np


def build_network(data_path="data"):
    n = pypsa.Network()

    buses = pd.read_csv(f"{data_path}/buses.csv")
    for _, b in buses.iterrows():
        n.add("Bus", b["name"], v_nom=b["v_nom"], carrier=b["carrier"])

    links = pd.read_csv(f"{data_path}/links.csv")
    for _, l in links.iterrows():
        n.add(
            "Link",
            f"{l['bus0']}-{l['bus1']}",
            bus0=l["bus0"],
            bus1=l["bus1"],
            p_nom=l["p_nom"],
            length=l["length"],
            carrier=l["carrier"],
        )

    gens = pd.read_csv(f"{data_path}/generators.csv")
    for _, g in gens.iterrows():
        if g["carrier"] == "onshore wind":
            # time-varying availability via p_max_pu
            pass
        n.add(
            "Generator",
            f"{g['carrier']}_{g['bus']}",
            bus=g["bus"],
            carrier=g["carrier"],
            p_nom=g["p_nom"],
            marginal_cost=g["marginal_cost"],
        )

    storage = pd.read_csv(f"{data_path}/storage.csv")
    for _, s in storage.iterrows():
        n.add(
            "StorageUnit",
            f"{s['carrier']}_{s['bus']}",
            bus=s["bus"],
            carrier=s["carrier"],
            p_nom=s["p_nom"],
            max_hours=s["e_nom"] / s["p_nom"],
            efficiency_dispatch=s["efficiency"],
            efficiency_store=s["efficiency"],
            marginal_cost=0.0,
        )

    demand = pd.read_csv(f"{data_path}/demand.csv")
    hours = demand.shape[0]
    n.set_snapshots(pd.date_range("2020-01-01", periods=hours, freq="H"))

    for bus in buses["name"]:
        if bus in demand.columns:
            n.add("Load", bus, bus=bus, p_set=demand[bus].values)
        else:
            n.add("Load", bus, bus=bus, p_set=0.0)

    wind_cf = pd.read_csv(f"{data_path}/wind_cf.csv")
    # align with snapshots
    wind_cf.index = n.snapshots

    # set p_max_pu for wind generators
    for gen in n.generators.index:
        if n.generators.at[gen, "carrier"] == "onshore wind":
            bus = n.generators.at[gen, "bus"]
            if bus in wind_cf.columns:
                n.generators_t.p_max_pu[gen] = wind_cf[bus]

    return n


def solve_and_export(n, outputs_path="../outputs"):
    n.optimize(solver_name="highs")

    # export main results
    n.generators_t.p.to_csv(f"{outputs_path}/generators_p.csv")
    n.storage_units_t.p.to_csv(f"{outputs_path}/storage_p.csv")
    n.loads_t.p.to_csv(f"{outputs_path}/loads_p.csv")
    n.buses_t.marginal_price.to_csv(f"{outputs_path}/nodal_prices.csv")


if __name__ == "__main__":
    network = build_network("data")
    solve_and_export(network, "outputs")
