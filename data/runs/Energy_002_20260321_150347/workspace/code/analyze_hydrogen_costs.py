#!/usr/bin/env python3
import csv
import json
import math
import os
from collections import defaultdict

WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(WORKDIR, 'data')
OUT = os.path.join(WORKDIR, 'outputs')
IMG = os.path.join(WORKDIR, 'report', 'images')

EUROPE_BENCHMARK = {
    'europe_cf': 0.42,
    'electrolyzer_capex_kgday': 950.0,
    'renewable_capex_kw': 1180.0,
    'fixed_opex_frac': 0.04,
    'electricity_buffer_usd_per_kg': 0.25,
    'distribution_usd_per_kg': 0.28,
}

SCENARIOS = [
    {
        'name': 'Base_low_rate',
        'label': 'Base / low-rate',
        'wacc': 0.06,
        'electrolyzer_capex_kgday': 700.0,
        'renewable_capex_kw': 900.0,
        'fixed_opex_frac': 0.03,
        'policy_credit_usd_per_kg': 0.0,
        'description': 'Favorable 2030 financing with moderate technology learning and no explicit de-risking credit.'
    },
    {
        'name': 'High_rate',
        'label': 'High-rate',
        'wacc': 0.12,
        'electrolyzer_capex_kgday': 700.0,
        'renewable_capex_kw': 900.0,
        'fixed_opex_frac': 0.03,
        'policy_credit_usd_per_kg': 0.0,
        'description': 'High global interest-rate environment without project de-risking.'
    },
    {
        'name': 'De_risked',
        'label': 'De-risked + policy',
        'wacc': 0.04,
        'electrolyzer_capex_kgday': 650.0,
        'renewable_capex_kw': 880.0,
        'fixed_opex_frac': 0.028,
        'policy_credit_usd_per_kg': 0.25,
        'description': 'Concessional / guaranteed finance plus policy support for export corridors.'
    }
]


def ensure_dirs():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(IMG, exist_ok=True)


def read_hexes(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            out = {'hex_id': row['hex_id']}
            for k, v in row.items():
                if k == 'hex_id':
                    continue
                out[k] = float(v)
            rows.append(out)
    return rows


def crf(rate, years):
    if rate == 0:
        return 1.0 / years
    return rate * (1 + rate) ** years / (((1 + rate) ** years) - 1)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def weighted_cf(site):
    # stylized hybrid renewable CF from solar and wind complementarity
    return clamp(0.58 * site['theo_pv'] + 0.42 * site['theo_wind'], 0.20, 0.90)


def desal_cost(site):
    # modest distance-linked water supply proxy in USD/kg-H2
    return 0.015 + 0.00055 * site['waterbody_dist_km']


def onsite_infra_cost(site):
    # proxy for road/grid spur + last-mile connectivity
    return (0.045 +
            0.0011 * site['road_dist_km'] +
            0.0015 * site['grid_dist_km'])


def port_link_cost(site):
    # inland transport / pipeline / transmission access to export port
    return 0.035 + 0.00125 * site['ocean_dist_km']


def electricity_lcoh_component(site, scenario):
    cf = weighted_cf(site)
    full_load_hours = cf * 8760.0
    kg_per_kw_year = full_load_hours / 52.0  # 52 kWh/kg-H2 in 2030
    annualized_renewable = scenario['renewable_capex_kw'] * (crf(scenario['wacc'], 25) + 0.02)
    renewable_usd_per_kg = annualized_renewable / max(kg_per_kw_year, 1e-6)
    balancing = 0.06 + 0.14 * (0.70 - cf if cf < 0.70 else 0.0)
    return renewable_usd_per_kg + balancing


def electrolyzer_lcoh_component(site, scenario):
    cf = weighted_cf(site)
    flh = cf * 8760.0
    kg_per_kgday_year = 365.0 * clamp(cf / 0.72, 0.45, 1.0)
    annualized = scenario['electrolyzer_capex_kgday'] * (crf(scenario['wacc'], 18) + scenario['fixed_opex_frac'])
    util_penalty = 0.18 * max(0.0, 0.62 - cf)
    return annualized / max(kg_per_kgday_year, 1e-6) + util_penalty


def h2_production_cost(site, scenario):
    return (electricity_lcoh_component(site, scenario) +
            electrolyzer_lcoh_component(site, scenario) +
            desal_cost(site) +
            onsite_infra_cost(site))


def ammonia_export_chain_cost(site, scenario):
    ammonia_synthesis = 0.52
    storage_loading = 0.14
    shipping = 0.22 + 0.00028 * site['ocean_dist_km']
    reconversion = 0.68
    terminal = 0.10
    corridor_efficiency = -0.08 if site['ocean_dist_km'] < 120 else 0.0
    return ammonia_synthesis + storage_loading + shipping + reconversion + terminal + port_link_cost(site) + corridor_efficiency - scenario['policy_credit_usd_per_kg']


def delivered_cost(site, scenario):
    prod = h2_production_cost(site, scenario)
    chain = ammonia_export_chain_cost(site, scenario)
    return prod + chain, prod, chain


def europe_domestic_cost(scenario):
    cf = EUROPE_BENCHMARK['europe_cf']
    flh = cf * 8760.0
    kg_per_kw_year = flh / 52.0
    annualized_renewable = EUROPE_BENCHMARK['renewable_capex_kw'] * (crf(scenario['wacc'], 25) + 0.025)
    renewable = annualized_renewable / kg_per_kw_year + EUROPE_BENCHMARK['electricity_buffer_usd_per_kg']
    annualized_el = EUROPE_BENCHMARK['electrolyzer_capex_kgday'] * (crf(scenario['wacc'], 18) + EUROPE_BENCHMARK['fixed_opex_frac'])
    electrolyzer = annualized_el / (365.0 * clamp(cf / 0.72, 0.45, 1.0)) + 0.10
    return renewable + electrolyzer + EUROPE_BENCHMARK['distribution_usd_per_kg']


def parse_dbf_records(dbf_path):
    with open(dbf_path, 'rb') as f:
        header = f.read(32)
        num_records = int.from_bytes(header[4:8], 'little')
        header_len = int.from_bytes(header[8:10], 'little')
        rec_len = int.from_bytes(header[10:12], 'little')
        num_fields = (header_len - 33) // 32
        fields = []
        for _ in range(num_fields):
            desc = f.read(32)
            name = desc[:11].split(b'\x00', 1)[0].decode('latin1').strip()
            ftype = chr(desc[11])
            flen = desc[16]
            fields.append((name, ftype, flen))
        f.read(1)  # terminator
        records = []
        for _ in range(num_records):
            rec = f.read(rec_len)
            if not rec or rec[0] == 0x2A:
                continue
            pos = 1
            row = {}
            for name, ftype, flen in fields:
                raw = rec[pos:pos + flen]
                pos += flen
                txt = raw.decode('latin1', errors='ignore').strip()
                row[name] = txt
            records.append(row)
    return records


def read_shp_polygons(shp_path):
    polys = []
    with open(shp_path, 'rb') as f:
        f.read(100)
        while True:
            rec_header = f.read(8)
            if not rec_header or len(rec_header) < 8:
                break
            rec_num = int.from_bytes(rec_header[:4], 'big')
            rec_len_words = int.from_bytes(rec_header[4:8], 'big')
            content = f.read(rec_len_words * 2)
            if len(content) < 4:
                break
            shape_type = int.from_bytes(content[:4], 'little')
            if shape_type not in (5, 3):
                continue
            pos = 4
            # bbox
            pos += 32
            num_parts = int.from_bytes(content[pos:pos + 4], 'little'); pos += 4
            num_points = int.from_bytes(content[pos:pos + 4], 'little'); pos += 4
            parts = [int.from_bytes(content[pos + i * 4: pos + (i + 1) * 4], 'little') for i in range(num_parts)]
            pos += 4 * num_parts
            points = []
            for _ in range(num_points):
                x = float.fromhex('0x0')
                x = int.from_bytes(content[pos:pos + 8], 'little', signed=False)
                y = int.from_bytes(content[pos + 8:pos + 16], 'little', signed=False)
                import struct
                px = struct.unpack('<d', content[pos:pos + 8])[0]
                py = struct.unpack('<d', content[pos + 8:pos + 16])[0]
                points.append((px, py))
                pos += 16
            parts_idx = parts + [len(points)]
            rings = []
            for i in range(len(parts)):
                ring = points[parts_idx[i]:parts_idx[i+1]]
                if len(ring) >= 3:
                    rings.append(ring)
            polys.append(rings)
    return polys


def mercator(lon, lat, width, height, bounds):
    min_lon, max_lon, min_lat, max_lat = bounds
    x = (lon - min_lon) / (max_lon - min_lon) * width
    lat = clamp(lat, -85, 85)
    min_m = math.log(math.tan(math.pi / 4 + math.radians(min_lat) / 2))
    max_m = math.log(math.tan(math.pi / 4 + math.radians(max_lat) / 2))
    ym = math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    y = height * (1 - (ym - min_m) / (max_m - min_m))
    return x, y


def svg_header(w, h):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">', '<rect width="100%" height="100%" fill="white"/>']


def svg_footer():
    return ['</svg>']


def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def save_svg(path, lines):
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def quantile(vals, q):
    vals = sorted(vals)
    if not vals:
        return 0.0
    i = (len(vals) - 1) * q
    lo = int(math.floor(i)); hi = int(math.ceil(i))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - i) + vals[hi] * (i - lo)


def draw_histogram(path, values, title, xlabel, bins=10, width=900, height=500, color='#4C78A8'):
    mn, mx = min(values), max(values)
    if mx == mn:
        mx = mn + 1.0
    counts = [0] * bins
    for v in values:
        idx = int((v - mn) / (mx - mn) * bins)
        if idx == bins:
            idx -= 1
        counts[idx] += 1
    margin = {'l': 70, 'r': 20, 't': 60, 'b': 60}
    plot_w = width - margin['l'] - margin['r']
    plot_h = height - margin['t'] - margin['b']
    ymax = max(counts) if counts else 1
    lines = svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="32" font-size="22" text-anchor="middle" font-family="Arial">{esc(title)}</text>')
    x0, y0 = margin['l'], height - margin['b']
    lines.append(f'<line x1="{x0}" y1="{margin["t"]}" x2="{x0}" y2="{y0}" stroke="black"/>')
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{width-margin["r"]}" y2="{y0}" stroke="black"/>')
    bw = plot_w / bins
    for i, c in enumerate(counts):
        h = plot_h * c / ymax
        x = x0 + i * bw
        y = y0 - h
        lines.append(f'<rect x="{x+1:.1f}" y="{y:.1f}" width="{bw-2:.1f}" height="{h:.1f}" fill="{color}"/>')
    lines.append(f'<text x="{width/2}" y="{height-18}" font-size="15" text-anchor="middle" font-family="Arial">{esc(xlabel)}</text>')
    save_svg(path, lines + svg_footer())


def draw_boxplot(path, groups, title, ylabel, width=900, height=520):
    margin = {'l': 80, 'r': 20, 't': 60, 'b': 80}
    plot_w = width - margin['l'] - margin['r']
    plot_h = height - margin['t'] - margin['b']
    all_vals = [v for _, vals in groups for v in vals]
    mn, mx = min(all_vals), max(all_vals)
    span = mx - mn if mx > mn else 1.0
    lines = svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="32" font-size="22" text-anchor="middle" font-family="Arial">{esc(title)}</text>')
    x0, y0 = margin['l'], height - margin['b']
    lines.append(f'<line x1="{x0}" y1="{margin["t"]}" x2="{x0}" y2="{y0}" stroke="black"/>')
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{width-margin["r"]}" y2="{y0}" stroke="black"/>')
    step = plot_w / len(groups)
    for i, (label, vals) in enumerate(groups):
        vals = sorted(vals)
        q1 = quantile(vals, 0.25); med = quantile(vals, 0.5); q3 = quantile(vals, 0.75)
        lo = min(vals); hi = max(vals)
        x = x0 + step * (i + 0.5)
        def yy(v):
            return y0 - (v - mn) / span * plot_h
        lines.append(f'<line x1="{x:.1f}" y1="{yy(lo):.1f}" x2="{x:.1f}" y2="{yy(hi):.1f}" stroke="#444"/>')
        lines.append(f'<rect x="{x-25:.1f}" y="{yy(q3):.1f}" width="50" height="{yy(q1)-yy(q3):.1f}" fill="#A0CBE8" stroke="#333"/>')
        lines.append(f'<line x1="{x-25:.1f}" y1="{yy(med):.1f}" x2="{x+25:.1f}" y2="{yy(med):.1f}" stroke="#333" stroke-width="2"/>')
        lines.append(f'<text x="{x:.1f}" y="{y0+22}" font-size="13" text-anchor="middle" font-family="Arial">{esc(label)}</text>')
    lines.append(f'<text transform="translate(20,{height/2}) rotate(-90)" font-size="15" text-anchor="middle" font-family="Arial">{esc(ylabel)}</text>')
    save_svg(path, lines + svg_footer())


def draw_scatter(path, pts, title, xlabel, ylabel, width=900, height=520):
    margin = {'l': 80, 'r': 30, 't': 60, 'b': 60}
    plot_w = width - margin['l'] - margin['r']
    plot_h = height - margin['t'] - margin['b']
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    xmin, xmax = min(xs), max(xs); ymin, ymax = min(ys), max(ys)
    padx = (xmax - xmin) * 0.06 or 1.0; pady = (ymax - ymin) * 0.08 or 1.0
    xmin -= padx; xmax += padx; ymin -= pady; ymax += pady
    def mp(x, y):
        px = margin['l'] + (x - xmin) / (xmax - xmin) * plot_w
        py = height - margin['b'] - (y - ymin) / (ymax - ymin) * plot_h
        return px, py
    lines = svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="32" font-size="22" text-anchor="middle" font-family="Arial">{esc(title)}</text>')
    lines.append(f'<line x1="{margin["l"]}" y1="{height-margin["b"]}" x2="{width-margin["r"]}" y2="{height-margin["b"]}" stroke="black"/>')
    lines.append(f'<line x1="{margin["l"]}" y1="{margin["t"]}" x2="{margin["l"]}" y2="{height-margin["b"]}" stroke="black"/>')
    for x, y, c in pts:
        px, py = mp(x, y)
        lines.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{c}" fill-opacity="0.75"/>')
    lines.append(f'<text x="{width/2}" y="{height-18}" font-size="15" text-anchor="middle" font-family="Arial">{esc(xlabel)}</text>')
    lines.append(f'<text transform="translate(20,{height/2}) rotate(-90)" font-size="15" text-anchor="middle" font-family="Arial">{esc(ylabel)}</text>')
    save_svg(path, lines + svg_footer())


def draw_bars(path, labels, values, title, ylabel, color='#F58518', width=900, height=520):
    margin = {'l': 80, 'r': 20, 't': 60, 'b': 120}
    plot_w = width - margin['l'] - margin['r']
    plot_h = height - margin['t'] - margin['b']
    vmax = max(values) if values else 1.0
    lines = svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="32" font-size="22" text-anchor="middle" font-family="Arial">{esc(title)}</text>')
    x0, y0 = margin['l'], height - margin['b']
    lines.append(f'<line x1="{x0}" y1="{margin["t"]}" x2="{x0}" y2="{y0}" stroke="black"/>')
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{width-margin["r"]}" y2="{y0}" stroke="black"/>')
    step = plot_w / len(values)
    bw = step * 0.65
    for i, v in enumerate(values):
        x = x0 + i * step + (step - bw) / 2
        h = plot_h * v / vmax
        y = y0 - h
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{h:.1f}" fill="{color}"/>')
        lines.append(f'<text transform="translate({x+bw/2:.1f},{y0+16}) rotate(40)" font-size="12" text-anchor="start" font-family="Arial">{esc(labels[i])}</text>')
        lines.append(f'<text x="{x+bw/2:.1f}" y="{y-6:.1f}" font-size="12" text-anchor="middle" font-family="Arial">{v:.2f}</text>')
    lines.append(f'<text transform="translate(20,{height/2}) rotate(-90)" font-size="15" text-anchor="middle" font-family="Arial">{esc(ylabel)}</text>')
    save_svg(path, lines + svg_footer())


def draw_map(path, africa_rings, sites, title, width=1000, height=780):
    bounds = (-20.0, 55.0, -36.0, 38.0)
    lines = svg_header(width, height)
    lines.append(f'<text x="{width/2}" y="30" font-size="22" text-anchor="middle" font-family="Arial">{esc(title)}</text>')
    lines.append('<rect x="50" y="50" width="900" height="680" fill="#eef6fb" stroke="#ccc"/>')
    for rings in africa_rings:
        for ring in rings:
            pts = []
            for lon, lat in ring:
                x, y = mercator(lon, lat, 900, 680, bounds)
                pts.append(f'{x+50:.1f},{y+50:.1f}')
            lines.append(f'<polygon points="{" ".join(pts)}" fill="#f3efe4" stroke="#b9b0a1" stroke-width="0.6"/>')
    costs = [s['delivered_cost'] for s in sites]
    mn, mx = min(costs), max(costs)
    for s in sites:
        x, y = mercator(s['lon'], s['lat'], 900, 680, bounds)
        t = 0 if mx == mn else (s['delivered_cost'] - mn) / (mx - mn)
        r = int(44 + 220 * t)
        g = int(160 - 90 * t)
        b = int(80 + 40 * (1 - t))
        lines.append(f'<circle cx="{x+50:.1f}" cy="{y+50:.1f}" r="5.5" fill="rgb({r},{g},{b})" fill-opacity="0.88" stroke="white" stroke-width="0.6"/>')
    lines.append(f'<text x="780" y="730" font-size="12" font-family="Arial">Low cost</text>')
    lines.append(f'<rect x="845" y="719" width="20" height="10" fill="rgb(44,160,120)"/>')
    lines.append(f'<rect x="865" y="719" width="20" height="10" fill="rgb(154,115,100)"/>')
    lines.append(f'<rect x="885" y="719" width="20" height="10" fill="rgb(264,70,80)"/>')
    lines.append(f'<text x="912" y="730" font-size="12" font-family="Arial">High cost</text>')
    save_svg(path, lines + svg_footer())


def top_n(rows, scenario_name, n=5):
    rr = sorted(rows, key=lambda r: r[f'{scenario_name}_delivered'])[:n]
    return rr


def summarize_results(sites, scenario_outputs, europe_costs):
    summary = {'n_sites': len(sites), 'scenarios': {}, 'europe_costs': europe_costs}
    for sc in SCENARIOS:
        name = sc['name']
        vals = [r[f'{name}_delivered'] for r in scenario_outputs]
        summary['scenarios'][name] = {
            'min_delivered': min(vals),
            'p25_delivered': quantile(vals, 0.25),
            'median_delivered': quantile(vals, 0.5),
            'p75_delivered': quantile(vals, 0.75),
            'max_delivered': max(vals),
            'competitive_vs_europe_count': sum(1 for r in scenario_outputs if r[f'{name}_delivered'] < europe_costs[name]),
            'top5': [
                {
                    'hex_id': r['hex_id'],
                    'lat': r['lat'],
                    'lon': r['lon'],
                    'delivered_cost': r[f'{name}_delivered'],
                    'production_cost': r[f'{name}_production'],
                    'export_chain_cost': r[f'{name}_chain']
                }
                for r in top_n(scenario_outputs, name, 5)
            ]
        }
    return summary


def main():
    ensure_dirs()
    sites = read_hexes(os.path.join(DATA, 'hex_final_NA_min.csv'))

    scenario_rows = []
    europe_costs = {}
    for site in sites:
        row = dict(site)
        for sc in SCENARIOS:
            total, prod, chain = delivered_cost(site, sc)
            row[f'{sc["name"]}_delivered'] = total
            row[f'{sc["name"]}_production'] = prod
            row[f'{sc["name"]}_chain'] = chain
            row[f'{sc["name"]}_competitiveness_gap_vs_europe'] = total - europe_domestic_cost(sc)
        scenario_rows.append(row)
    for sc in SCENARIOS:
        europe_costs[sc['name']] = europe_domestic_cost(sc)

    with open(os.path.join(OUT, 'site_costs.csv'), 'w', newline='', encoding='utf-8') as f:
        fieldnames = list(scenario_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(scenario_rows)

    summary = summarize_results(sites, scenario_rows, europe_costs)
    with open(os.path.join(OUT, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    draw_histogram(
        os.path.join(IMG, 'figure_1_resource_histogram.svg'),
        [weighted_cf(s) for s in sites],
        'Distribution of hybrid renewable capacity factor across candidate African sites',
        'Hybrid renewable capacity factor'
    )

    draw_boxplot(
        os.path.join(IMG, 'figure_2_scenario_boxplot.svg'),
        [(sc['label'], [r[f'{sc["name"]}_delivered'] for r in scenario_rows]) for sc in SCENARIOS],
        'Delivered African hydrogen cost by financing / policy scenario',
        'Delivered cost to Europe (USD/kg-H2)'
    )

    pts = []
    for r in scenario_rows:
        gap = r['Base_low_rate_competitiveness_gap_vs_europe']
        color = '#2ca02c' if gap < 0 else '#d62728'
        pts.append((weighted_cf(r), r['Base_low_rate_delivered'], color))
    draw_scatter(
        os.path.join(IMG, 'figure_3_resource_vs_cost.svg'),
        pts,
        'Resource quality versus delivered cost (base / low-rate scenario)',
        'Hybrid renewable capacity factor',
        'Delivered cost to Europe (USD/kg-H2)'
    )

    labels = [sc['label'] for sc in SCENARIOS]
    values = [europe_costs[sc['name']] for sc in SCENARIOS]
    draw_bars(
        os.path.join(IMG, 'figure_4_europe_benchmark.svg'),
        labels,
        values,
        'Stylized European domestic green hydrogen cost benchmark by scenario',
        'USD/kg-H2'
    )

    # build Africa basemap from shapefile/dbf, keep Africa only
    dbf_records = parse_dbf_records(os.path.join(DATA, 'africa_map', 'ne_10m_admin_0_countries.dbf'))
    shp_polys = read_shp_polygons(os.path.join(DATA, 'africa_map', 'ne_10m_admin_0_countries.shp'))
    africa_polys = []
    for rec, rings in zip(dbf_records, shp_polys):
        continent = rec.get('CONTINENT', '')
        if continent == 'Africa':
            africa_polys.append(rings)

    base_sites = []
    for r in scenario_rows:
        base_sites.append({'lon': r['lon'], 'lat': r['lat'], 'delivered_cost': r['Base_low_rate_delivered']})
    draw_map(
        os.path.join(IMG, 'figure_5_cost_map.svg'),
        africa_polys,
        base_sites,
        'African candidate sites: delivered cost map (base / low-rate scenario)'
    )

    # top-site comparison chart
    top_base = sorted(scenario_rows, key=lambda r: r['Base_low_rate_delivered'])[:8]
    draw_bars(
        os.path.join(IMG, 'figure_6_top_sites.svg'),
        [r['hex_id'] for r in top_base],
        [r['Base_low_rate_delivered'] for r in top_base],
        'Eight least-cost export sites in Africa (base / low-rate scenario)',
        'Delivered cost to Europe (USD/kg-H2)',
        color='#54A24B'
    )

    # scenario competitiveness summary table
    with open(os.path.join(OUT, 'europe_benchmark.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['scenario', 'europe_domestic_cost_usd_per_kg'])
        for sc in SCENARIOS:
            w.writerow([sc['name'], round(europe_costs[sc['name']], 6)])

    with open(os.path.join(OUT, 'top_sites_by_scenario.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['scenario', 'rank', 'hex_id', 'lat', 'lon', 'delivered_cost_usd_per_kg', 'production_cost_usd_per_kg', 'export_chain_cost_usd_per_kg'])
        for sc in SCENARIOS:
            name = sc['name']
            for i, r in enumerate(sorted(scenario_rows, key=lambda x: x[f'{name}_delivered'])[:10], start=1):
                w.writerow([name, i, r['hex_id'], round(r['lat'], 6), round(r['lon'], 6), round(r[f'{name}_delivered'], 6), round(r[f'{name}_production'], 6), round(r[f'{name}_chain'], 6)])

    print('Processed', len(sites), 'sites')
    for sc in SCENARIOS:
        name = sc['name']
        vals = [r[f'{name}_delivered'] for r in scenario_rows]
        print(name, 'min', round(min(vals), 3), 'median', round(quantile(vals, 0.5), 3), 'europe', round(europe_costs[name], 3), 'competitive', sum(1 for r in scenario_rows if r[f'{name}_delivered'] < europe_costs[name]))

if __name__ == '__main__':
    main()
