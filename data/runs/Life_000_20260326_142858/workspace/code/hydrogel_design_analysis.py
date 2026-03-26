import zipfile, xml.etree.ElementTree as ET, json, math, statistics, random, zlib, struct
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'
OUT = BASE / 'outputs'
IMG = BASE / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
random.seed(7)
NS={'a':'http://schemas.openxmlformats.org/spreadsheetml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
MONOMERS=['Nucleophilic-HEA','Hydrophobic-BA','Acidic-CBEA','Cationic-ATAC','Aromatic-PEA','Amide-AAm']
TARGET='Glass (kPa)_10s'
TARGET2='Glass (kPa)_60s'
OPT_TARGET='Glass (kPa)_max'

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


def load_xlsx(path):
    z=zipfile.ZipFile(path)
    wb=ET.fromstring(z.read('xl/workbook.xml'))
    rels=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
    relmap={r.attrib['Id']:r.attrib['Target'] for r in rels}
    sst=[]
    if 'xl/sharedStrings.xml' in z.namelist():
        root=ET.fromstring(z.read('xl/sharedStrings.xml'))
        for si in root:
            txt=[]
            for t in si.iter('{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t'):
                txt.append(t.text or '')
            sst.append(''.join(txt))
    out={}
    for sh in wb.find('a:sheets',NS):
        name=sh.attrib['name']
        rid=sh.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
        ws=ET.fromstring(z.read('xl/'+relmap[rid]))
        rows=ws.find('a:sheetData',NS)
        data=[]
        for r in rows:
            vals=[]
            for c in r:
                v=c.find('a:v',NS)
                val=v.text if v is not None else ''
                if c.attrib.get('t')=='s' and val!='': val=sst[int(val)]
                vals.append(val)
            if any(x!='' for x in vals): data.append(vals)
        out[name]=data
    return out


def parse_float(x):
    if x in ('', '/', None): return None
    try: return float(x)
    except: return None


def rows_to_dicts(data):
    header=data[0]
    out=[]
    for row in data[1:]:
        if len(row)<len(header): row=row+['']*(len(header)-len(row))
        d={header[i]:row[i] for i in range(len(header))}
        out.append(d)
    return out


def load_datasets():
    train=rows_to_dicts(load_xlsx(DATA/'184_verified_Original Data_ML_20230926.xlsx')['Data_to_HU'])
    final_wb=load_xlsx(DATA/'ML_ei&pred (1&2&3rounds)_20240408.xlsx')
    ei=rows_to_dicts(final_wb['EI'])
    pred=rows_to_dicts(final_wb['PRED'])
    return train, ei, pred


def get_xy(rows, target_col):
    X=[]; y=[]
    for r in rows:
        vals=[parse_float(r[m]) for m in MONOMERS]
        t=parse_float(r.get(target_col))
        if all(v is not None for v in vals) and t is not None:
            X.append(vals)
            y.append(t)
    return X,y


def normalize(X):
    d=len(X[0])
    means=[sum(r[j] for r in X)/len(X) for j in range(d)]
    stds=[]
    for j in range(d):
        v=sum((r[j]-means[j])**2 for r in X)/len(X)
        stds.append(math.sqrt(v) if v>1e-12 else 1.0)
    Xn=[[ (r[j]-means[j])/stds[j] for j in range(d)] for r in X]
    return Xn,means,stds


def apply_norm(X,means,stds):
    return [[(r[j]-means[j])/stds[j] for j in range(len(r))] for r in X]


def fit_linear(X,y,epochs=600,lr=0.04):
    d=len(X[0]); w=[0.0]*d; b=0.0
    for _ in range(epochs):
        gw=[0.0]*d; gb=0.0
        for x,yy in zip(X,y):
            pred=sum(w[j]*x[j] for j in range(d))+b
            err=pred-yy
            gb+=err
            for j in range(d): gw[j]+=err*x[j]
        n=len(X)
        for j in range(d): w[j]-=lr*(gw[j]/n + 1e-4*w[j])
        b-=lr*(gb/n)
    return w,b


def predict_linear(X,w,b):
    return [sum(w[j]*x[j] for j in range(len(w)))+b for x in X]


def rmse(y,p):
    return math.sqrt(sum((a-b)**2 for a,b in zip(y,p))/len(y))

def mae(y,p):
    return sum(abs(a-b) for a,b in zip(y,p))/len(y)

def r2(y,p):
    ym=sum(y)/len(y)
    ssr=sum((a-b)**2 for a,b in zip(y,p))
    sst=sum((a-ym)**2 for a in y)
    return 1-ssr/sst if sst>0 else 0.0

def corr(a,b):
    am=sum(a)/len(a); bm=sum(b)/len(b)
    num=sum((x-am)*(y-bm) for x,y in zip(a,b))
    da=math.sqrt(sum((x-am)**2 for x in a)); db=math.sqrt(sum((y-bm)**2 for y in b))
    return num/(da*db) if da>0 and db>0 else 0.0


def split_idx(n):
    idx=list(range(n)); random.Random(7).shuffle(idx)
    nt=max(1,int(0.8*n))
    return idx[:nt], idx[nt:]


def train_model(X,y):
    tr,te=split_idx(len(X))
    Xtr=[X[i] for i in tr]; ytr=[y[i] for i in tr]
    Xte=[X[i] for i in te]; yte=[y[i] for i in te]
    Xtrn,m,s=normalize(Xtr)
    Xten=apply_norm(Xte,m,s)
    w,b=fit_linear(Xtrn,ytr)
    ptr=predict_linear(Xtrn,w,b); pte=predict_linear(Xten,w,b)
    return {'means':m,'stds':s,'w':w,'b':b,
            'train_rmse':rmse(ytr,ptr),'test_rmse':rmse(yte,pte),
            'train_r2':r2(ytr,ptr),'test_r2':r2(yte,pte),
            'test_mae':mae(yte,pte), 'test_corr':corr(yte,pte)}


def predict_formula(comp, model):
    x=apply_norm([comp], model['means'], model['stds'])[0]
    return sum(model['w'][j]*x[j] for j in range(len(x)))+model['b']


def design_candidates(model, n=30000):
    cand=[]
    for _ in range(n):
        raw=[random.random() for _ in MONOMERS]
        s=sum(raw)
        comp=[x/s for x in raw]
        pred=predict_formula(comp, model)
        cand.append((pred, comp))
    cand.sort(key=lambda x:x[0], reverse=True)
    return cand[:200]


def summarize(train, ei, pred, model10, model60):
    X,y10=get_xy(train,TARGET)
    _,y60=get_xy(train,TARGET2)
    best_train=max(y10)
    above_1mpa=sum(1 for y in y10 if y>=1000)
    ei_rows=[]
    for r in ei:
        vals=[parse_float(r[m]) for m in MONOMERS]
        t=parse_float(r.get(OPT_TARGET))
        if all(v is not None for v in vals) and t is not None:
            ei_rows.append((vals,t))
    top_ei=max(t for _,t in ei_rows) if ei_rows else None
    top_pred=max(parse_float(r.get(OPT_TARGET)) for r in pred if parse_float(r.get(OPT_TARGET)) is not None)
    cands=design_candidates(model10)
    top_designs=[]
    for score, comp in cands[:10]:
        score60=predict_formula(comp, model60)
        top_designs.append({'pred_glass10_kpa':score,'pred_glass60_kpa':score60, **{m:v for m,v in zip(MONOMERS,comp)}})
    avg_top={m:sum(d[m] for d in top_designs)/len(top_designs) for m in MONOMERS}
    return {
        'n_train': len(y10),
        'best_train_glass10_kpa': best_train,
        'fraction_above_1MPa_train': above_1mpa/len(y10),
        'best_ei_glassmax_kpa': top_ei,
        'best_predicted_candidate_kpa': top_pred,
        'model_10s': model10,
        'model_60s': model60,
        'top_de_novo_designs': top_designs,
        'avg_top_design_composition': avg_top,
    }


def plot_data_overview(train_y, ei_y, path):
    c=Canvas(1000,520)
    x0,y0,x1,y1=70,40,950,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    vals=[len(train_y), len(ei_y)]
    vmax=max(vals)
    for i,v in enumerate(vals):
        left=180+i*300; right=320+i*300
        top=y1-int(v/vmax*(y1-y0-40))
        c.fillrect(left,top,right,y1-1,[(90,150,220),(220,160,90)][i])
    c.save_png(path)


def plot_main_results(summary, path):
    vals=[summary['best_train_glass10_kpa'], summary['best_ei_glassmax_kpa'], summary['best_predicted_candidate_kpa'], summary['top_de_novo_designs'][0]['pred_glass10_kpa']]
    c=Canvas(1000,520)
    x0,y0,x1,y1=70,40,950,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    vmax=max(vals)
    cols=[(100,150,220),(220,130,80),(100,180,120),(170,100,220)]
    for i,v in enumerate(vals):
        left=120+i*190; right=210+i*190
        top=y1-int(v/vmax*(y1-y0-40))
        c.fillrect(left,top,right,y1-1,cols[i])
        if v>=1000:
            c.line(left, y1-int(1000/vmax*(y1-y0-40)), right, y1-int(1000/vmax*(y1-y0-40)), (0,0,0), 1)
    c.save_png(path)


def plot_validation(summary, path):
    vals=[summary['model_10s']['test_r2'], summary['model_10s']['test_corr'], summary['model_60s']['test_r2'], summary['model_60s']['test_corr']]
    c=Canvas(1000,520)
    x0,y0,x1,y1=70,40,950,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    for i,v in enumerate(vals):
        left=120+i*190; right=210+i*190
        top=y1-int(max(0,v)*(y1-y0-40))
        c.fillrect(left,top,right,y1-1,(120,100+25*i,220-25*i))
    c.save_png(path)


def plot_feature_importance(model, path):
    vals=[abs(x) for x in model['w']]
    vmax=max(vals)
    c=Canvas(1000,520)
    x0,y0,x1,y1=70,40,950,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    for i,v in enumerate(vals):
        left=120+i*120; right=200+i*120
        top=y1-int(v/vmax*(y1-y0-40))
        c.fillrect(left,top,right,y1-1,(140,100,220))
    c.save_png(path)


def plot_top_design(summary, path):
    d=summary['top_de_novo_designs'][0]
    vals=[d[m] for m in MONOMERS]
    c=Canvas(1000,520)
    x0,y0,x1,y1=70,40,950,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    vmax=max(vals)
    for i,v in enumerate(vals):
        left=110+i*130; right=190+i*130
        top=y1-int(v/vmax*(y1-y0-40))
        c.fillrect(left,top,right,y1-1,(90,170,130))
    c.save_png(path)


def main():
    train, ei, pred = load_datasets()
    X10,y10=get_xy(train,TARGET)
    X60,y60=get_xy(train,TARGET2)
    model10=train_model(X10,y10)
    model60=train_model(X60,y60)
    summary=summarize(train, ei, pred, model10, model60)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with open(OUT/'top_designs.csv','w',encoding='utf-8') as f:
        f.write('rank,pred_glass10_kpa,pred_glass60_kpa,'+','.join(MONOMERS)+'\n')
        for i,d in enumerate(summary['top_de_novo_designs'],1):
            f.write(f"{i},{d['pred_glass10_kpa']},{d['pred_glass60_kpa']}," + ','.join(str(d[m]) for m in MONOMERS) + '\n')
    with open(OUT/'model_metrics.csv','w',encoding='utf-8') as f:
        f.write('model,test_r2,test_corr,test_rmse,test_mae\n')
        f.write(f"glass10,{model10['test_r2']},{model10['test_corr']},{model10['test_rmse']},{model10['test_mae']}\n")
        f.write(f"glass60,{model60['test_r2']},{model60['test_corr']},{model60['test_rmse']},{model60['test_mae']}\n")
    ei_y=[parse_float(r.get(OPT_TARGET)) for r in ei if parse_float(r.get(OPT_TARGET)) is not None]
    plot_data_overview(y10, ei_y, IMG/'data_overview.png')
    plot_main_results(summary, IMG/'main_results.png')
    plot_validation(summary, IMG/'validation_comparison.png')
    plot_feature_importance(model10, IMG/'feature_importance.png')
    plot_top_design(summary, IMG/'top_design_composition.png')
    print(json.dumps(summary, indent=2))

if __name__=='__main__':
    main()
