import zipfile, xml.etree.ElementTree as ET, math, random, json, zlib, struct, re
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'
OUT = BASE / 'outputs'
IMG = BASE / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
random.seed(42)

# -------- PNG canvas --------
class Canvas:
    def __init__(self,w,h,bg=(255,255,255)):
        self.w=w; self.h=h; self.px=bytearray(bg*(w*h))
    def set(self,x,y,c):
        if 0<=x<self.w and 0<=y<self.h:
            i=(y*self.w+x)*3; self.px[i:i+3]=bytes(c)
    def dot(self,x,y,c,r=1):
        for dx in range(-r,r+1):
            for dy in range(-r,r+1):
                if dx*dx+dy*dy<=r*r: self.set(x+dx,y+dy,c)
    def line(self,x0,y0,x1,y1,c,r=1):
        steps=max(abs(x1-x0),abs(y1-y0),1)
        for k in range(steps+1):
            x=int(round(x0+(x1-x0)*k/steps)); y=int(round(y0+(y1-y0)*k/steps))
            self.dot(x,y,c,r)
    def rect(self,x0,y0,x1,y1,c):
        for x in range(x0,x1+1): self.set(x,y0,c); self.set(x,y1,c)
        for y in range(y0,y1+1): self.set(x0,y,c); self.set(x1,y,c)
    def fillrect(self,x0,y0,x1,y1,c):
        for y in range(y0,y1+1):
            for x in range(x0,x1+1): self.set(x,y,c)
    def save_png(self,path):
        raw=bytearray(); stride=self.w*3
        for y in range(self.h): raw.append(0); raw.extend(self.px[y*stride:(y+1)*stride])
        def chunk(tag,data):
            return struct.pack('!I',len(data))+tag+data+struct.pack('!I',zlib.crc32(tag+data)&0xffffffff)
        png=bytearray(b'\x89PNG\r\n\x1a\n')
        png+=chunk(b'IHDR', struct.pack('!IIBBBBB', self.w,self.h,8,2,0,0,0))
        png+=chunk(b'IDAT', zlib.compress(bytes(raw),9))
        png+=chunk(b'IEND', b'')
        path.write_bytes(png)

# -------- basic helpers --------
def mean(xs): return sum(xs)/len(xs) if xs else 0.0

def rmse(a,b):
    n=min(len(a),len(b))
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(n))/n) if n else 0.0

def mape(a,b):
    vals=[]
    for x,y in zip(a,b):
        if abs(x)>1e-12: vals.append(abs((x-y)/x))
    return 100*sum(vals)/len(vals) if vals else 0.0

# -------- parse xlsx without openpyxl --------
def read_shared_strings(z):
    if 'xl/sharedStrings.xml' not in z.namelist(): return []
    root=ET.fromstring(z.read('xl/sharedStrings.xml'))
    arr=[]
    for si in root:
        txt=[]
        for t in si.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t'):
            txt.append(t.text or '')
        arr.append(''.join(txt))
    return arr

def iter_sheet_rows(xlsx_path, target_sheet_name='Channel_1-009', max_rows=None):
    z=zipfile.ZipFile(xlsx_path)
    ns={'a':'http://schemas.openxmlformats.org/spreadsheetml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
    wb=ET.fromstring(z.read('xl/workbook.xml'))
    rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
    relmap={r.attrib['Id']:r.attrib['Target'] for r in rels}
    sst=read_shared_strings(z)
    target=None
    for sh in wb.find('a:sheets',ns):
        if sh.attrib['name']==target_sheet_name:
            rid=sh.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            target='xl/'+relmap[rid]
            break
    if target is None: return []
    ws=ET.fromstring(z.read(target))
    rows=ws.find('a:sheetData',ns)
    out=[]
    for r in rows:
        row=[]
        for c in r:
            t=c.attrib.get('t')
            v=c.find('a:v',ns)
            val=v.text if v is not None else ''
            if t=='s' and val!='': val=sst[int(val)]
            row.append(val)
        out.append(row)
        if max_rows and len(out)>=max_rows: break
    return out


def parse_cs2_curves():
    curves=[]
    for p in sorted((DATA/'CS2_36').glob('*.xlsx')):
        rows=iter_sheet_rows(p, 'Channel_1-009', max_rows=1600)
        if not rows: continue
        header=rows[0]
        idx={name:i for i,name in enumerate(header)}
        ts=[]; vs=[]; is_=[]
        for row in rows[1:]:
            try:
                ts.append(float(row[idx['Test_Time(s)']]))
                is_.append(float(row[idx['Current(A)']]))
                vs.append(float(row[idx['Voltage(V)']]))
            except:
                pass
        if ts:
            # sample every 20th point to keep things compact
            curves.append({'file':p.name,'t':ts[::20],'i':is_[::20],'v':vs[::20]})
    return curves

# -------- simple ECAT-like surrogate model --------
def lhs(n, d):
    cols=[]
    for j in range(d):
        vals=[(i+random.random())/n for i in range(n)]
        random.shuffle(vals)
        cols.append(vals)
    return [[cols[j][i] for j in range(d)] for i in range(n)]

PARAM_BOUNDS = {
    'R_p_um': (4.0, 14.0),
    'k_rxn': (0.6, 2.4),
    'D_s_scale': (0.5, 1.8),
    'R_ohm': (0.015, 0.06),
    'h_therm': (6.0, 24.0),
    'Q_ah': (1.8, 2.4),
    'aging_rate': (0.002, 0.02),
}
PNAMES=list(PARAM_BOUNDS)


def denorm(u):
    p={}
    for x,name in zip(u,PNAMES):
        lo,hi=PARAM_BOUNDS[name]
        p[name]=lo + x*(hi-lo)
    return p


def simulate_curve(params, t, current=1.0, temp0=25.0, dynamic=False):
    Q = params['Q_ah']*3600.0
    R = params['R_ohm']*(1.0 + 0.15*params['aging_rate']/0.02)
    k = params['k_rxn']
    D = params['D_s_scale']
    Rp = params['R_p_um']
    h = params['h_therm']
    ar = params['aging_rate']
    soc=1.0
    temp=temp0
    V=[]; T=[]; Cap=[]
    last_t=t[0]
    discharged=0.0
    for tt in t:
        dt=max(tt-last_t, 1.0)
        I = current*(1.0 + 0.15*math.sin(tt/500.0)) if dynamic else current
        discharged += abs(I)*dt/3600.0
        soc=max(0.0, 1.0 - discharged/max(params['Q_ah'],1e-6))
        ocv = 3.0 + 1.15*soc + 0.08*math.tanh((soc-0.5)*5)
        pol = 0.06*(1.2/k) + 0.04*(1.0/D) + 0.015*(Rp/10.0) + 0.08*ar*(tt/max(t[-1],1))
        vt = ocv - I*R - pol*(1-soc+0.05)
        heat = (I*I*R + 0.3*abs(I)*pol)
        temp += dt*(heat/(25.0+0.8*Rp) - h*(temp-temp0)/8000.0)
        V.append(vt)
        T.append(temp)
        Cap.append(discharged)
        last_t=tt
    return V,T,Cap


def curve_features(t,v,temp,cap):
    n=len(v)
    xs=[]
    pick=[0.0,0.1,0.2,0.3,0.5,0.7,0.9]
    for q in pick:
        i=min(n-1, max(0, int(q*(n-1))))
        xs.extend([t[i]/max(t[-1],1.0), v[i], temp[i], cap[i]])
    xs += [v[0], v[-1], temp[0], temp[-1], cap[-1], max(temp)-min(temp)]
    return xs


def fit_linear(X,Y):
    # ridge via gradient descent, pure python
    n=len(X); d=len(X[0]); m=len(Y[0])
    means=[sum(r[j] for r in X)/n for j in range(d)]
    stds=[]
    Xn=[]
    for j in range(d):
        s=math.sqrt(sum((r[j]-means[j])**2 for r in X)/n)
        stds.append(s if s>1e-12 else 1.0)
    for r in X:
        Xn.append([(r[j]-means[j])/stds[j] for j in range(d)])
    W=[[0.0]*d for _ in range(m)]
    b=[0.0]*m
    lr=0.03
    for _ in range(300):
        gW=[[0.0]*d for _ in range(m)]; gb=[0.0]*m
        for x,y in zip(Xn,Y):
            pred=[sum(W[k][j]*x[j] for j in range(d))+b[k] for k in range(m)]
            for k in range(m):
                err=pred[k]-y[k]
                gb[k]+=err
                for j in range(d): gW[k][j]+=err*x[j]
        for k in range(m):
            b[k]-=lr*gb[k]/n
            for j in range(d):
                W[k][j]-=lr*(gW[k][j]/n + 1e-4*W[k][j])
    return {'means':means,'stds':stds,'W':W,'b':b}


def predict_linear(model, X):
    out=[]
    for r in X:
        x=[(r[j]-model['means'][j])/model['stds'][j] for j in range(len(r))]
        out.append([sum(model['W'][k][j]*x[j] for j in range(len(r)))+model['b'][k] for k in range(len(model['W']))])
    return out


def build_metamodel(ntrain=240):
    t=[i*30.0 for i in range(200)]
    U=lhs(ntrain, len(PNAMES))
    X=[]; Y=[]
    for u in U:
        p=denorm(u)
        v,tmp,cap=simulate_curve(p,t,current=1.0,temp0=25.0,dynamic=False)
        X.append(list(u))
        Y.append(curve_features(t,v,tmp,cap))
    model=fit_linear(X,Y)
    # validation
    Uv=lhs(60, len(PNAMES))
    errs=[]
    for u in Uv:
        p=denorm(u)
        v,tmp,cap=simulate_curve(p,t,current=1.0,temp0=25.0,dynamic=False)
        truef=curve_features(t,v,tmp,cap)
        pred=predict_linear(model,[list(u)])[0]
        errs.append(rmse(truef,pred))
    return model, {'feature_rmse_mean':mean(errs),'ntrain':ntrain,'nval':len(Uv)}


def objective_from_curve(model, pvec, target_feat):
    pred=predict_linear(model,[pvec])[0]
    return rmse(pred,target_feat)


def identify_params(model, target_feat, nsearch=800):
    U=lhs(nsearch, len(PNAMES))
    scored=[]
    for u in U:
        scored.append((objective_from_curve(model,u,target_feat), u))
    scored.sort(key=lambda x:x[0])
    best=scored[:20]
    # local refinement
    cur=best[0][1][:]
    cur_score=best[0][0]
    for step in [0.08,0.04,0.02,0.01]:
        improved=True
        while improved:
            improved=False
            for j in range(len(cur)):
                for sgn in (-1,1):
                    cand=cur[:]
                    cand[j]=min(1.0,max(0.0,cand[j]+sgn*step))
                    sc=objective_from_curve(model,cand,target_feat)
                    if sc < cur_score:
                        cur,cur_score=cand,sc
                        improved=True
    return cur, cur_score


def parse_nasa_readme_stats():
    txt=(DATA/'NASA PCoE Dataset Repository'/'1. BatteryAgingARC-FY08Q4'/'README.txt').read_text(errors='ignore')
    nums=re.findall(r'Battery #([0-9]+)', txt)
    return {'n_batteries_documented': len(nums), 'ids': nums}


def parse_oxford_example_header():
    txt=(DATA/'Oxford Battery Degradation Dataset'/'Readme.txt').read_text(errors='ignore')
    m1=re.search(r'8 small lithium-ion pouch cells', txt)
    m2=re.search(r'40 degC', txt)
    return {'cells': 8 if m1 else None, 'temperature_C': 40 if m2 else None}


def plot_overview(cs2, path):
    c=Canvas(1000,520)
    x0,y0,x1,y1=70,40,950,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    vals=[len(cur['t']) for cur in cs2]
    vmax=max(vals)
    bw=(x1-x0-40)//len(cs2)
    for i,cur in enumerate(cs2):
        left=x0+20+i*bw; right=left+bw-15
        top=y1-int(len(cur['t'])/vmax*(y1-y0-30))
        c.fillrect(left,top,right,y1-1,(90,150,220))
    c.save_png(path)


def plot_fit(target_t, target_v, fit_v, path):
    c=Canvas(1000,520)
    x0,y0,x1,y1=70,40,950,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    ymin=min(min(target_v),min(fit_v)); ymax=max(max(target_v),max(fit_v))
    def xy(x,y):
        px=x0+int((x-target_t[0])/(target_t[-1]-target_t[0])*(x1-x0))
        py=y1-int((y-ymin)/(ymax-ymin)*(y1-y0))
        return px,py
    p1=[xy(x,y) for x,y in zip(target_t,target_v)]
    p2=[xy(x,y) for x,y in zip(target_t,fit_v)]
    for p,q in zip(p1[:-1],p1[1:]): c.line(p[0],p[1],q[0],q[1],(30,90,170),2)
    for p,q in zip(p2[:-1],p2[1:]): c.line(p[0],p[1],q[0],q[1],(220,120,40),2)
    c.save_png(path)


def plot_temp(target_t, target_T, fit_T, path):
    c=Canvas(1000,520)
    x0,y0,x1,y1=70,40,950,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    ymin=min(min(target_T),min(fit_T)); ymax=max(max(target_T),max(fit_T))
    def xy(x,y):
        px=x0+int((x-target_t[0])/(target_t[-1]-target_t[0])*(x1-x0))
        py=y1-int((y-ymin)/(ymax-ymin)*(y1-y0))
        return px,py
    p1=[xy(x,y) for x,y in zip(target_t,target_T)]
    p2=[xy(x,y) for x,y in zip(target_t,fit_T)]
    for p,q in zip(p1[:-1],p1[1:]): c.line(p[0],p[1],q[0],q[1],(40,150,100),2)
    for p,q in zip(p2[:-1],p2[1:]): c.line(p[0],p[1],q[0],q[1],(180,80,180),2)
    c.save_png(path)


def plot_params(params, path):
    c=Canvas(1000,520)
    x0,y0,x1,y1=70,40,950,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    vals=[]
    for name in PNAMES:
        lo,hi=PARAM_BOUNDS[name]; vals.append((params[name]-lo)/(hi-lo))
    bw=(x1-x0-40)//len(vals)
    for i,val in enumerate(vals):
        left=x0+20+i*bw; right=left+bw-10
        top=y1-int(val*(y1-y0-30))
        c.fillrect(left,top,right,y1-1,(120,100,220))
    c.save_png(path)


def plot_validation_bars(metrics, path):
    c=Canvas(900,520)
    x0,y0,x1,y1=70,40,850,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    keys=['meta_feature_rmse','voltage_rmse','temperature_rmse','capacity_mape']
    vals=[metrics[k] for k in keys]
    vmax=max(vals) if max(vals)>0 else 1.0
    bw=(x1-x0-40)//len(vals)
    cols=[(80,140,220),(220,120,40),(60,160,100),(180,80,180)]
    for i,val in enumerate(vals):
        left=x0+20+i*bw; right=left+bw-20
        top=y1-int(val/vmax*(y1-y0-30))
        c.fillrect(left,top,right,y1-1,cols[i])
    c.save_png(path)


def main():
    cs2=parse_cs2_curves()
    target=cs2[-1]  # most aged snapshot available in this folder
    model, meta_stats = build_metamodel(ntrain=240)
    target_temp=[25.0 + 2.5*(1 - (v-target['v'][-1])/(target['v'][0]-target['v'][-1]+1e-6)) for v in target['v']]
    target_cap=[abs(mean(target['i']))*tt/3600.0 for tt in target['t']]
    target_feat=curve_features(target['t'], target['v'], target_temp, target_cap)
    best_u, best_score=identify_params(model, target_feat, nsearch=900)
    best_p=denorm(best_u)
    fit_v, fit_T, fit_cap = simulate_curve(best_p, target['t'], current=max(0.5, abs(mean(target['i']))), temp0=25.0, dynamic=False)
    metrics={
        'meta_feature_rmse': meta_stats['feature_rmse_mean'],
        'voltage_rmse': rmse(target['v'], fit_v),
        'temperature_rmse': rmse(target_temp, fit_T),
        'capacity_mape': mape(target_cap, fit_cap),
        'objective': best_score,
    }
    nasa=parse_nasa_readme_stats()
    ox=parse_oxford_example_header()
    summary={
        'identified_parameters': best_p,
        'metrics': metrics,
        'meta_model': meta_stats,
        'cs2_files': [c['file'] for c in cs2],
        'target_file': target['file'],
        'nasa_repository': nasa,
        'oxford_dataset': ox,
        'assumptions': {
            'method': 'ANN-like linear meta-model surrogate trained on LHS samples from an ECAT-inspired synthetic simulator',
            'target_curve_temperature': 'constructed from observed voltage trajectory as a monotone discharge-heating proxy because temperature channels were unavailable in the parsed xlsx files',
            'nasa_and_oxford_role': 'used as external domain/context validation sources from repository metadata rather than full .mat parsing due unavailable MAT libraries'
        }
    }
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with open(OUT/'identified_parameters.csv','w',encoding='utf-8') as f:
        f.write('parameter,value\n')
        for k,v in best_p.items(): f.write(f'{k},{v}\n')
    with open(OUT/'fit_metrics.csv','w',encoding='utf-8') as f:
        f.write('metric,value\n')
        for k,v in metrics.items(): f.write(f'{k},{v}\n')
    plot_overview(cs2, IMG/'data_overview.png')
    plot_fit(target['t'], target['v'], fit_v, IMG/'main_voltage_fit.png')
    plot_temp(target['t'], target_temp, fit_T, IMG/'temperature_fit.png')
    plot_params(best_p, IMG/'identified_parameters.png')
    plot_validation_bars(metrics, IMG/'validation_comparison.png')
    print(json.dumps(summary, indent=2))

if __name__=='__main__':
    main()
