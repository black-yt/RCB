import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'outputs'
REPORT_IMG_DIR = ROOT / 'report' / 'images'


def ensure_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    REPORT_IMG_DIR.mkdir(exist_ok=True, parents=True)


def load_inputs():
    buses = pd.read_csv(DATA_DIR / 'buses.csv')
    links = pd.read_csv(DATA_DIR / 'links.csv')
    generators = pd.read_csv(DATA_DIR / 'generators.csv')
    storage = pd.read_csv(DATA_DIR / 'storage.csv')
    demand = pd.read_csv(DATA_DIR / 'demand.csv')
    wind_cf = pd.read_csv(DATA_DIR / 'wind_cf.csv')
    return buses, links, generators, storage, demand, wind_cf


def prepare_components(generators, storage):
    generators = generators.copy()
    generators['gen_id'] = ['G%03d' % i for i in range(len(generators))]
    storage = storage.copy()
    storage['store_id'] = ['S%02d' % i for i in range(len(storage))]
    return generators, storage


def build_incidence_matrix(buses, links):
    bus_names = list(buses['name'])
    bus_idx = {b: i for i, b in enumerate(bus_names)}
    A = np.zeros((len(bus_names), len(links)))
    for l, row in links.iterrows():
        A[bus_idx[row['bus0']], l] = -1.0
        A[bus_idx[row['bus1']], l] = 1.0
    return A, bus_idx, bus_names


def scenario_config():
    return {
        'baseline': {
            'description': 'Observed one-week demand and wind availability with storage enabled.',
            'demand_scale': 1.0,
            'wind_scale': 1.0,
            'gas_cost_add': 0.0,
            'storage_enabled': True,
        },
        'high_demand': {
            'description': 'Stylized stress test with 15% higher demand.',
            'demand_scale': 1.15,
            'wind_scale': 1.0,
            'gas_cost_add': 0.0,
            'storage_enabled': True,
        },
        'low_wind': {
            'description': 'Stylized stress test with 35% lower wind availability.',
            'demand_scale': 1.0,
            'wind_scale': 0.65,
            'gas_cost_add': 0.0,
            'storage_enabled': True,
        },
        'no_storage': {
            'description': 'Flexibility sensitivity disabling storage.',
            'demand_scale': 1.0,
            'wind_scale': 1.0,
            'gas_cost_add': 0.0,
            'storage_enabled': False,
        },
        'gas_price_shock': {
            'description': 'Fuel-price stress test adding 30 GBP/MWh to gas marginal cost.',
            'demand_scale': 1.0,
            'wind_scale': 1.0,
            'gas_cost_add': 30.0,
            'storage_enabled': True,
        },
    }


def get_generator_availability(generators, wind_cf, scenario):
    T = len(wind_cf)
    G = len(generators)
    avail = np.zeros((T, G))
    for g, row in generators.iterrows():
        if row['carrier'] == 'onshore wind':
            cf = wind_cf[row['bus']].to_numpy() * scenario['wind_scale']
            cf = np.clip(cf, 0.0, 1.0)
            avail[:, g] = row['p_nom'] * cf
        else:
            avail[:, g] = row['p_nom']
    return avail


def run_scenario(name, scenario, buses, links, generators, storage, demand, wind_cf):
    generators = generators.copy()
    storage = storage.copy()
    if scenario['gas_cost_add'] != 0:
        generators.loc[generators['carrier'] == 'gas', 'marginal_cost'] += scenario['gas_cost_add']
    if not scenario['storage_enabled']:
        storage = storage.iloc[0:0].copy()

    demand_s = demand.copy() * scenario['demand_scale']
    gen_avail = get_generator_availability(generators, wind_cf, scenario)

    A, bus_idx, bus_names = build_incidence_matrix(buses, links)
    T = len(demand_s)
    B = len(bus_names)
    G = len(generators)
    L = len(links)
    S = len(storage)

    n_gen = T * G
    n_flow = T * L
    n_charge = T * S
    n_discharge = T * S
    n_soc = T * S
    n_shed = T * B
    n_total = n_gen + n_flow + n_charge + n_discharge + n_soc + n_shed

    def gen_idx(t, g):
        return t * G + g

    def flow_idx(t, l):
        return n_gen + t * L + l

    def charge_idx(t, s):
        return n_gen + n_flow + t * S + s

    def discharge_idx(t, s):
        return n_gen + n_flow + n_charge + t * S + s

    def soc_idx(t, s):
        return n_gen + n_flow + n_charge + n_discharge + t * S + s

    def shed_idx(t, b):
        return n_gen + n_flow + n_charge + n_discharge + n_soc + t * B + b

    c = np.zeros(n_total)
    for t in range(T):
        for g, row in generators.iterrows():
            c[gen_idx(t, g)] = row['marginal_cost']
        for b in range(B):
            c[shed_idx(t, b)] = 5000.0

    bounds = []
    for t in range(T):
        for g in range(G):
            bounds.append((0.0, float(gen_avail[t, g])))
    for t in range(T):
        for _, row in links.iterrows():
            cap = float(row['p_nom'])
            bounds.append((-cap, cap))
    for t in range(T):
        for _, row in storage.iterrows():
            bounds.append((0.0, float(row['p_nom'])))
    for t in range(T):
        for _, row in storage.iterrows():
            bounds.append((0.0, float(row['p_nom'])))
    for t in range(T):
        for _, row in storage.iterrows():
            bounds.append((0.0, float(row['e_nom'])))
    for t in range(T):
        for b in range(B):
            bounds.append((0.0, float(demand_s.iloc[t, b])))

    A_eq = []
    b_eq = []

    gen_by_bus = {b: [] for b in bus_names}
    for g, row in generators.iterrows():
        gen_by_bus[row['bus']].append(g)

    store_by_bus = {b: [] for b in bus_names}
    for s, row in storage.iterrows():
        store_by_bus[row['bus']].append(s)

    for t in range(T):
        for b, bus in enumerate(bus_names):
            row = np.zeros(n_total)
            for g in gen_by_bus[bus]:
                row[gen_idx(t, g)] = 1.0
            for l in range(L):
                row[flow_idx(t, l)] = A[b, l]
            for s in store_by_bus[bus]:
                row[charge_idx(t, s)] = -1.0
                row[discharge_idx(t, s)] = 1.0
            row[shed_idx(t, b)] = 1.0
            A_eq.append(row)
            b_eq.append(float(demand_s.iloc[t, b]))

    for s, row_s in storage.iterrows():
        eff = float(row_s['efficiency']) ** 0.5
        init_soc = 0.5 * float(row_s['e_nom'])
        for t in range(T):
            row = np.zeros(n_total)
            row[soc_idx(t, s)] = 1.0
            if t > 0:
                row[soc_idx(t - 1, s)] = -1.0
                rhs = 0.0
            else:
                rhs = init_soc
            row[charge_idx(t, s)] = -eff
            row[discharge_idx(t, s)] = 1.0 / eff
            A_eq.append(row)
            b_eq.append(rhs)
        cyc = np.zeros(n_total)
        cyc[soc_idx(T - 1, s)] = 1.0
        A_eq.append(cyc)
        b_eq.append(init_soc)

    result = linprog(
        c=c,
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method='highs',
    )
    if not result.success:
        raise RuntimeError(f'Scenario {name} failed: {result.message}')

    x = result.x
    gen_dispatch = np.zeros((T, G))
    flows = np.zeros((T, L))
    charge = np.zeros((T, S)) if S > 0 else np.zeros((T, 0))
    discharge = np.zeros((T, S)) if S > 0 else np.zeros((T, 0))
    soc = np.zeros((T, S)) if S > 0 else np.zeros((T, 0))
    shed = np.zeros((T, B))
    for t in range(T):
        for g in range(G):
            gen_dispatch[t, g] = x[gen_idx(t, g)]
        for l in range(L):
            flows[t, l] = x[flow_idx(t, l)]
        for s in range(S):
            charge[t, s] = x[charge_idx(t, s)]
            discharge[t, s] = x[discharge_idx(t, s)]
            soc[t, s] = x[soc_idx(t, s)]
        for b in range(B):
            shed[t, b] = x[shed_idx(t, b)]

    dispatch_df = pd.DataFrame(gen_dispatch, columns=generators['gen_id'])
    gen_meta = generators.set_index('gen_id')
    dispatch_by_carrier = pd.DataFrame({
        carrier: dispatch_df.loc[:, gen_meta[gen_meta['carrier'] == carrier].index].sum(axis=1)
        for carrier in generators['carrier'].unique()
    })
    available_wind = gen_meta[gen_meta['carrier'] == 'onshore wind']['p_nom']
    wind_gen_ids = list(available_wind.index)
    wind_available_ts = pd.DataFrame(gen_avail[:, gen_meta['carrier'].to_numpy() == 'onshore wind'], columns=wind_gen_ids)
    wind_dispatch_ts = dispatch_df[wind_gen_ids]
    curtailment = wind_available_ts.sum(axis=1) - wind_dispatch_ts.sum(axis=1)

    flow_abs = np.abs(flows)
    link_util = flow_abs / links['p_nom'].to_numpy()[None, :]

    summary = {
        'scenario': name,
        'description': scenario['description'],
        'objective_gbp': float(result.fun),
        'total_demand_mwh': float(demand_s.sum().sum()),
        'served_mwh': float(demand_s.sum().sum() - shed.sum()),
        'unserved_mwh': float(shed.sum()),
        'unserved_share': float(shed.sum() / demand_s.sum().sum()),
        'wind_generation_mwh': float(wind_dispatch_ts.sum().sum()),
        'wind_available_mwh': float(wind_available_ts.sum().sum()),
        'wind_curtailment_mwh': float(curtailment.sum()),
        'wind_curtailment_share': float(curtailment.sum() / max(wind_available_ts.sum().sum(), 1e-9)),
        'gas_generation_mwh': float(dispatch_by_carrier.get('gas', pd.Series(dtype=float)).sum() if 'gas' in dispatch_by_carrier else 0.0),
        'nuclear_generation_mwh': float(dispatch_by_carrier.get('nuclear', pd.Series(dtype=float)).sum() if 'nuclear' in dispatch_by_carrier else 0.0),
        'storage_discharge_mwh': float(discharge.sum()),
        'storage_charge_mwh': float(charge.sum()),
        'max_line_utilization': float(link_util.max()),
        'mean_line_utilization': float(link_util.mean()),
        'peak_unserved_mw': float(shed.sum(axis=1).max()),
    }

    metrics = pd.DataFrame({
        'hour': np.arange(T),
        'demand_mw': demand_s.sum(axis=1),
        'wind_mw': wind_dispatch_ts.sum(axis=1),
        'gas_mw': dispatch_by_carrier['gas'] if 'gas' in dispatch_by_carrier else 0.0,
        'nuclear_mw': dispatch_by_carrier['nuclear'] if 'nuclear' in dispatch_by_carrier else 0.0,
        'storage_discharge_mw': discharge.sum(axis=1) if S > 0 else 0.0,
        'storage_charge_mw': charge.sum(axis=1) if S > 0 else 0.0,
        'curtailment_mw': curtailment,
        'unserved_mw': shed.sum(axis=1),
        'mean_line_utilization': link_util.mean(axis=1),
        'max_line_utilization': link_util.max(axis=1),
    })

    bus_balance = pd.DataFrame(index=np.arange(T))
    for b, bus in enumerate(bus_names):
        bus_balance[bus] = demand_s[bus] - shed[:, b]

    artifacts = {
        'summary': summary,
        'metrics': metrics,
        'carrier_dispatch': dispatch_by_carrier,
        'line_flows': pd.DataFrame(flows, columns=[f"{r.bus0}->{r.bus1}" for _, r in links.iterrows()]),
        'line_utilization': pd.DataFrame(link_util, columns=[f"{r.bus0}->{r.bus1}" for _, r in links.iterrows()]),
        'storage_soc': pd.DataFrame(soc, columns=storage['store_id']) if S > 0 else pd.DataFrame(index=np.arange(T)),
        'storage_charge': pd.DataFrame(charge, columns=storage['store_id']) if S > 0 else pd.DataFrame(index=np.arange(T)),
        'storage_discharge': pd.DataFrame(discharge, columns=storage['store_id']) if S > 0 else pd.DataFrame(index=np.arange(T)),
        'wind_curtailment': pd.DataFrame({'hour': np.arange(T), 'curtailment_mw': curtailment}),
        'dispatch_generators': dispatch_df,
    }
    return artifacts


def save_outputs(name, artifacts):
    scenario_dir = OUTPUT_DIR / name
    scenario_dir.mkdir(exist_ok=True, parents=True)
    with open(scenario_dir / 'summary.json', 'w') as f:
        json.dump(artifacts['summary'], f, indent=2)
    artifacts['metrics'].to_csv(scenario_dir / 'metrics.csv', index=False)
    artifacts['carrier_dispatch'].to_csv(scenario_dir / 'carrier_dispatch.csv', index=False)
    artifacts['line_flows'].to_csv(scenario_dir / 'line_flows.csv', index=False)
    artifacts['line_utilization'].to_csv(scenario_dir / 'line_utilization.csv', index=False)
    artifacts['wind_curtailment'].to_csv(scenario_dir / 'wind_curtailment.csv', index=False)
    artifacts['dispatch_generators'].to_csv(scenario_dir / 'dispatch_generators.csv', index=False)
    if not artifacts['storage_soc'].empty:
        artifacts['storage_soc'].to_csv(scenario_dir / 'storage_soc.csv', index=False)
        artifacts['storage_charge'].to_csv(scenario_dir / 'storage_charge.csv', index=False)
        artifacts['storage_discharge'].to_csv(scenario_dir / 'storage_discharge.csv', index=False)


def create_summary_table(all_summaries):
    df = pd.DataFrame(all_summaries)
    df['cost_per_mwh_served'] = df['objective_gbp'] / df['served_mwh']
    df.to_csv(OUTPUT_DIR / 'scenario_summary.csv', index=False)
    return df


def main():
    ensure_dirs()
    buses, links, generators, storage, demand, wind_cf = load_inputs()
    generators, storage = prepare_components(generators, storage)

    config = scenario_config()
    all_summaries = []
    for name, sc in config.items():
        artifacts = run_scenario(name, sc, buses, links, generators, storage, demand, wind_cf)
        save_outputs(name, artifacts)
        all_summaries.append(artifacts['summary'])
    summary_df = create_summary_table(all_summaries)
    print(summary_df[['scenario', 'objective_gbp', 'unserved_mwh', 'wind_curtailment_share', 'max_line_utilization']].to_string(index=False))


if __name__ == '__main__':
    main()
