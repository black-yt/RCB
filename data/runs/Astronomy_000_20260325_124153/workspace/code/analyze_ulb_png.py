import math
import json
import zlib
import struct
import statistics
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'
OUT = BASE / 'outputs'
IMG = BASE / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

G = 6.67430e-11
c = 299792458.0
hbar = 1.054571817e-34
M_sun = 1.98847e30
EV_PER_J = 1.0 / 1.602176634e-19
T_HUBBLE = 13.8e9 * 365.25 * 24 * 3600
MU_CONST = hbar * c**3 / G * EV_PER_J


def q(xs, p):
    ys = sorted(xs)
    i = (len(ys)-1) * p
    lo = int(i)
    hi = min(lo + 1, len(ys)-1)
    t = i - lo
    return ys[lo]*(1-t) + ys[hi]*t


def load_samples(path):
    mass, spin = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            m, a = map(float, s.split()[:2])
            mass.append(m)
            spin.append(a)
    return mass, spin


def omega_h_dimless(a):
    return a / (2.0 * (1.0 + math.sqrt(max(0.0, 1.0 - a*a))))


def spin_threshold(alpha):
    if alpha <= 0:
        return 0.0
    if alpha >= 0.499999:
        return 0.999999
    lo, hi = 0.0, 0.999999
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if omega_h_dimless(mid) < alpha:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)

THRESH_CACHE = [spin_threshold(i/1000.0) for i in range(500)]

def spin_threshold_fast(alpha):
    if alpha <= 0:
        return 0.0
    if alpha >= 0.499:
        return 0.999999
    x = alpha * 1000.0
    i = int(x)
    t = x - i
    return THRESH_CACHE[i] * (1-t) + THRESH_CACHE[i+1] * t


def tau_sr_seconds(alpha, spin, mass_msun):
    rg = G * mass_msun * M_sun / c**3
    aeff = max(spin - spin_threshold_fast(alpha), 0.0)
    if aeff <= 0.0 or alpha <= 0.0:
        return 1e300
    gamma = (aeff * alpha**9) / (24.0 * rg)
    return 1.0 / gamma if gamma > 0 else 1e300


def exclusion_prob_for_mu(masses, spins, mu_ev, tau_cut=T_HUBBLE):
    count = 0
    alpha_sum = 0.0
    n = len(masses)
    for M, a in zip(masses, spins):
        alpha = mu_ev * (M * M_sun) / MU_CONST
        alpha_sum += alpha
        if alpha <= 0.0 or alpha >= 0.499:
            continue
        if a > spin_threshold_fast(alpha) and tau_sr_seconds(alpha, a, M) < tau_cut:
            count += 1
    return count / n, alpha_sum / n


def coupling_cap(mu_ev, mass_msun):
    m_cloud = 0.1 * mass_msun * M_sun
    mu_si = mu_ev / EV_PER_J / (c**2)
    fmin = math.sqrt(max(m_cloud * mu_si * c**2 / 10.0, 1e-60))
    return min((mu_si / fmin)**2, 1.0)


def infer_lambda_limit(masses, mu_grid, probs):
    med_m = q(masses, 0.5)
    vals = []
    for mu, p in zip(mu_grid, probs):
        if p >= 0.95:
            vals.append((mu, coupling_cap(mu, med_m)))
    if not vals:
        return None
    mu_best, lam_best = min(vals, key=lambda x: x[1])
    return {'mu_at_95': mu_best, 'lambda_upper_proxy': lam_best}

class Canvas:
    def __init__(self, w, h, bg=(255,255,255)):
        self.w, self.h = w, h
        self.px = bytearray(bg * (w*h))
    def set(self, x, y, color):
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y*self.w + x)*3
            self.px[i:i+3] = bytes(color)
    def dot(self, x, y, color, r=1):
        for dx in range(-r, r+1):
            for dy in range(-r, r+1):
                if dx*dx + dy*dy <= r*r:
                    self.set(x+dx, y+dy, color)
    def line(self, x0, y0, x1, y1, color, r=1):
        steps = max(abs(x1-x0), abs(y1-y0), 1)
        for k in range(steps+1):
            x = int(round(x0 + (x1-x0)*k/steps))
            y = int(round(y0 + (y1-y0)*k/steps))
            self.dot(x, y, color, r=r)
    def rect(self, x0, y0, x1, y1, color):
        for x in range(x0, x1+1):
            self.set(x, y0, color); self.set(x, y1, color)
        for y in range(y0, y1+1):
            self.set(x0, y, color); self.set(x1, y, color)
    def save_png(self, path):
        raw = bytearray()
        stride = self.w*3
        for y in range(self.h):
            raw.append(0)
            raw.extend(self.px[y*stride:(y+1)*stride])
        def chunk(tag, data):
            return struct.pack('!I', len(data)) + tag + data + struct.pack('!I', zlib.crc32(tag+data) & 0xffffffff)
        png = bytearray(b'\x89PNG\r\n\x1a\n')
        png += chunk(b'IHDR', struct.pack('!IIBBBBB', self.w, self.h, 8, 2, 0, 0, 0))
        png += chunk(b'IDAT', zlib.compress(bytes(raw), 9))
        png += chunk(b'IEND', b'')
        path.write_bytes(png)


def plot_scatter_two_panels(path, left, right):
    w, h = 1200, 520
    c = Canvas(w, h)
    panels = [(50, 40, 550, 470, left), (650, 40, 1150, 470, right)]
    axis_col=(0,0,0); pt_col=(40,90,160)
    for x0,y0,x1,y1,(xs,ys) in panels:
        c.rect(x0,y0,x1,y1,axis_col)
        xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys)
        for xv,yv in zip(xs,ys):
            px = x0 + 10 + int((xv-xmin)/(xmax-xmin) * (x1-x0-20))
            py = y1 - 10 - int((yv-ymin)/(ymax-ymin) * (y1-y0-20))
            c.dot(px, py, pt_col, r=1)
    c.save_png(path)


def plot_logx_lines(path, series_list, y_min=None, y_max=None, threshold=None):
    w, h = 900, 520
    c = Canvas(w, h)
    x0,y0,x1,y1 = 70,40,860,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    allx=[x for s in series_list for x,_ in s]
    ally=[y for s in series_list for _,y in s]
    xmin,xmax=min(allx),max(allx)
    ymin=min(ally) if y_min is None else y_min
    ymax=max(ally) if y_max is None else y_max
    cols=[(40,90,160),(220,120,20),(60,160,90)]
    def mapxy(x,y):
        px=x0+int((math.log10(x)-math.log10(xmin))/(math.log10(xmax)-math.log10(xmin))*(x1-x0))
        py=y1-int((y-ymin)/(ymax-ymin)*(y1-y0)) if ymax>ymin else (y0+y1)//2
        return px,py
    if threshold is not None:
        a,b=mapxy(xmin,threshold),mapxy(xmax,threshold)
        c.line(a[0],a[1],b[0],b[1],(140,140,140),r=1)
    for idx,series in enumerate(series_list):
        pts=[mapxy(x,y) for x,y in series]
        for p,q in zip(pts[:-1],pts[1:]):
            c.line(p[0],p[1],q[0],q[1],cols[idx%len(cols)],r=1)
    c.save_png(path)


def main():
    datasets = {
        'M33 X-7': DATA / 'M33_X-7_samples.dat',
        'IRAS 09149-6206': DATA / 'IRAS_09149-6206_samples.dat',
    }
    mu_grid = [10**(-21 + i*(8/220)) for i in range(221)]
    results = {}
    summary = {}
    for name, path in datasets.items():
        masses, spins = load_samples(path)
        probs, alphas = [], []
        for mu in mu_grid:
            p, a_mean = exclusion_prob_for_mu(masses, spins, mu)
            probs.append(p)
            alphas.append(a_mean)
        lam_limit = infer_lambda_limit(masses, mu_grid, probs)
        exc = [mu for mu, p in zip(mu_grid, probs) if p >= 0.95]
        results[name] = {'masses': masses, 'spins': spins, 'mu_grid': mu_grid, 'excl_prob': probs, 'alpha_mean': alphas, 'lambda_limit': lam_limit}
        summary[name] = {
            'n_samples': len(masses),
            'mass_mean': statistics.fmean(masses),
            'mass_q05': q(masses, 0.05),
            'mass_q50': q(masses, 0.5),
            'mass_q95': q(masses, 0.95),
            'spin_mean': statistics.fmean(spins),
            'spin_q05': q(spins, 0.05),
            'spin_q50': q(spins, 0.5),
            'spin_q95': q(spins, 0.95),
            'mu_95_min': min(exc) if exc else None,
            'mu_95_max': max(exc) if exc else None,
            'lambda_limit': lam_limit,
        }
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with open(OUT/'exclusion_curves.csv', 'w', encoding='utf-8') as f:
        f.write('dataset,mu_ev,exclusion_probability,alpha_mean\n')
        for name, res in results.items():
            for mu,p,a in zip(res['mu_grid'], res['excl_prob'], res['alpha_mean']):
                f.write(f'{name},{mu:.12e},{p:.8f},{a:.8e}\n')
    plot_scatter_two_panels(IMG/'data_overview.png', (results['M33 X-7']['masses'], results['M33 X-7']['spins']), (results['IRAS 09149-6206']['masses'], results['IRAS 09149-6206']['spins']))
    plot_logx_lines(IMG/'exclusion_probability.png', [list(zip(results['M33 X-7']['mu_grid'], results['M33 X-7']['excl_prob'])), list(zip(results['IRAS 09149-6206']['mu_grid'], results['IRAS 09149-6206']['excl_prob']))], y_min=0.0, y_max=1.0, threshold=0.95)
    plot_logx_lines(IMG/'alpha_mapping.png', [list(zip(results['M33 X-7']['mu_grid'], results['M33 X-7']['alpha_mean'])), list(zip(results['IRAS 09149-6206']['mu_grid'], results['IRAS 09149-6206']['alpha_mean']))])
    lam_series=[]
    for name in ['M33 X-7','IRAS 09149-6206']:
        res=results[name]
        med_m=q(res['masses'],0.5)
        pts=[(mu, coupling_cap(mu, med_m)) for mu,p in zip(res['mu_grid'],res['excl_prob']) if p>=0.5]
        if pts:
            logpts=[(x, math.log10(y) if y>0 else -300) for x,y in pts]
            lam_series.append(logpts)
    plot_logx_lines(IMG/'self_interaction_limits.png', lam_series)
    # validation/comparison figure: exclusion probability around thresholds only
    focused=[]
    for name in ['M33 X-7','IRAS 09149-6206']:
        res=results[name]
        pts=[(mu,p) for mu,p in zip(res['mu_grid'],res['excl_prob']) if p>0.85]
        focused.append(pts if pts else list(zip(res['mu_grid'],res['excl_prob'])))
    plot_logx_lines(IMG/'validation_comparison.png', focused, y_min=0.85, y_max=1.0, threshold=0.95)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
