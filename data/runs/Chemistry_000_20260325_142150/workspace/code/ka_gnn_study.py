import csv, math, json, random, zlib, struct, re, statistics
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data'
OUT = BASE / 'outputs'
IMG = BASE / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)
random.seed(7)

ATOMS = ['B','C','N','O','F','P','S','Cl','Br','I']
ATOM_INDEX = {a:i for i,a in enumerate(ATOMS)}
TASK_FILES = ['bace.csv','bbbp.csv','clintox.csv','hiv.csv','muv.csv']
MAX_ROWS = {'bace.csv':1513,'bbbp.csv':2039,'clintox.csv':1477,'hiv.csv':4000,'muv.csv':6000}


def parse_smiles_tokens(smiles):
    tokens=[]
    i=0
    while i < len(smiles):
        ch=smiles[i]
        if ch == '[':
            j=smiles.find(']', i)
            if j == -1:
                atom = smiles[i+1:]
                i = len(smiles)
            else:
                atom = smiles[i+1:j]
                i = j+1
            m = re.search(r'([A-Z][a-z]?)', atom)
            if m:
                tokens.append(m.group(1))
            continue
        if i+1 < len(smiles) and smiles[i:i+2] in ('Cl','Br'):
            tokens.append(smiles[i:i+2]); i += 2; continue
        if ch.isalpha():
            if ch.islower():
                tokens.append(ch.upper())
            else:
                tokens.append(ch)
        i += 1
    return [t for t in tokens if t and t[0].isalpha()]


def smiles_features(smiles):
    toks = parse_smiles_tokens(smiles)
    total = len(toks)
    counts = [0]*len(ATOMS)
    aromatic = sum(1 for ch in smiles if ch.islower())
    ring_digits = sum(ch.isdigit() for ch in smiles)
    branches = smiles.count('(') + smiles.count(')')
    double_bonds = smiles.count('=')
    triple_bonds = smiles.count('#')
    charges = smiles.count('+') + smiles.count('-')
    halogens = 0
    hetero = 0
    for t in toks:
        if t in ATOM_INDEX:
            counts[ATOM_INDEX[t]] += 1
        if t in ('F','Cl','Br','I'):
            halogens += 1
        if t not in ('C','H'):
            hetero += 1
    heavy = sum(counts)
    if heavy == 0:
        heavy = total
    non_covalent_proxy = counts[ATOM_INDEX['N']] + counts[ATOM_INDEX['O']] + counts[ATOM_INDEX['S']] + halogens + charges
    fourier = [math.sin((k+1)*(heavy+1)/7.0) for k in range(8)] + [math.cos((k+1)*(hetero+1)/7.0) for k in range(8)]
    graph_stats = [heavy, hetero, halogens, aromatic, ring_digits, branches, double_bonds, triple_bonds, charges, non_covalent_proxy]
    return counts + graph_stats + fourier


def sigmoid(x):
    if x >= 0:
        z = math.exp(-x)
        return 1.0/(1.0+z)
    z = math.exp(x)
    return z/(1.0+z)


def auc_score(y, s):
    pairs = [(score, label) for label, score in zip(y, s) if label in (0,1)]
    if not pairs:
        return None
    pos = sum(1 for _,lab in pairs if lab==1)
    neg = sum(1 for _,lab in pairs if lab==0)
    if pos == 0 or neg == 0:
        return None
    pairs.sort(key=lambda x: x[0])
    rank_sum = 0.0
    i = 0
    rank = 1
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j-i) - 1) / 2.0
        pos_in_group = sum(1 for k in range(i,j) if pairs[k][1] == 1)
        rank_sum += pos_in_group * avg_rank
        rank += (j-i)
        i = j
    return (rank_sum - pos*(pos+1)/2.0) / (pos*neg)


def pr_auc(y, s):
    pairs = sorted([(score,label) for label,score in zip(y,s) if label in (0,1)], reverse=True)
    pos = sum(1 for _,lab in pairs if lab==1)
    if pos == 0:
        return None
    tp=fp=0
    prev_recall=0.0
    area=0.0
    for score,label in pairs:
        if label==1: tp += 1
        else: fp += 1
        recall = tp / pos
        precision = tp / (tp + fp)
        area += precision * (recall - prev_recall)
        prev_recall = recall
    return area


def balanced_accuracy(y, s):
    yy=[lab for lab in y if lab in (0,1)]
    ss=[score for lab,score in zip(y,s) if lab in (0,1)]
    if not yy: return None
    pred=[1 if x>=0.5 else 0 for x in ss]
    tp=sum(1 for a,b in zip(yy,pred) if a==1 and b==1)
    tn=sum(1 for a,b in zip(yy,pred) if a==0 and b==0)
    fp=sum(1 for a,b in zip(yy,pred) if a==0 and b==1)
    fn=sum(1 for a,b in zip(yy,pred) if a==1 and b==0)
    tpr=tp/(tp+fn) if tp+fn else 0.0
    tnr=tn/(tn+fp) if tn+fp else 0.0
    return 0.5*(tpr+tnr)


def avg(lst):
    lst=[x for x in lst if x is not None]
    return sum(lst)/len(lst) if lst else None


def parse_dataset(path):
    rows=[]
    with open(path, newline='', encoding='utf-8') as f:
        reader=csv.reader(f)
        header=next(reader)
        name=path.name
        if name=='bace.csv':
            s_idx=header.index('smiles'); tasks=[header.index('label')]
        elif name=='bbbp.csv':
            s_idx=header.index('smiles'); tasks=[header.index('label')]
        elif name=='clintox.csv':
            s_idx=header.index('smiles'); tasks=[header.index('FDA_APPROVED'), header.index('CT_TOX')]
        elif name=='hiv.csv':
            s_idx=header.index('smiles'); tasks=[header.index('label')]
        elif name=='muv.csv':
            s_idx=header.index('smiles') if 'smiles' in header else len(header)-1
            tasks=[i for i,h in enumerate(header) if h.startswith('MUV-')]
        else:
            raise ValueError(name)
        max_rows = MAX_ROWS.get(name)
        for ridx,row in enumerate(reader):
            if max_rows is not None and ridx >= max_rows:
                break
            if not row: continue
            if name=='muv.csv' and len(row) < len(header):
                row += [''] * (len(header)-len(row))
            smiles=row[s_idx].strip()
            if not smiles: continue
            labels=[]
            for ti in tasks:
                if ti >= len(row):
                    labels.append(None); continue
                val=row[ti].strip()
                if val == '': labels.append(None)
                else:
                    try: labels.append(int(float(val)))
                    except: labels.append(None)
            rows.append((smiles, labels))
    return rows


def split_rows(rows):
    idx=list(range(len(rows)))
    random.Random(7).shuffle(idx)
    n=len(idx)
    n_train=max(1, int(0.7*n)); n_val=max(1, int(0.15*n))
    train=idx[:n_train]; val=idx[n_train:n_train+n_val]; test=idx[n_train+n_val:]
    if not test: test=val; val=train[:max(1,len(train)//5)]
    return train,val,test


def normalize(trainX, X):
    d=len(trainX[0])
    means=[sum(row[j] for row in trainX)/len(trainX) for j in range(d)]
    stds=[]
    for j in range(d):
        v=sum((row[j]-means[j])**2 for row in trainX)/len(trainX)
        stds.append(math.sqrt(v) if v>1e-12 else 1.0)
    out=[]
    for row in X:
        out.append([(row[j]-means[j])/stds[j] for j in range(d)])
    return out, means, stds


def train_logreg(X, y, epochs=40, lr=0.05, l2=1e-4):
    d=len(X[0])
    w=[0.0]*d; b=0.0
    valid=[i for i,v in enumerate(y) if v in (0,1)]
    pos=sum(1 for i in valid if y[i]==1); neg=sum(1 for i in valid if y[i]==0)
    if pos==0 or neg==0: return w,b
    pos_w=neg/max(pos,1)
    for _ in range(epochs):
        gw=[0.0]*d; gb=0.0
        for i in valid:
            z=sum(w[j]*X[i][j] for j in range(d))+b
            p=sigmoid(z)
            yi=y[i]
            wt=pos_w if yi==1 else 1.0
            err=(p-yi)*wt
            for j in range(d): gw[j]+=err*X[i][j]
            gb+=err
        n=len(valid)
        for j in range(d):
            w[j]-=lr*(gw[j]/n + l2*w[j])
        b-=lr*(gb/n)
    return w,b


def predict(X,w,b):
    return [sigmoid(sum(w[j]*row[j] for j in range(len(w)))+b) for row in X]


def evaluate_task(X, labels, idxs, kind='ka'):
    feats=[]
    for row in X:
        if kind=='baseline':
            feats.append(row[:20])
        else:
            feats.append(row)
    tr,va,te=idxs
    trainX=[feats[i] for i in tr]
    allX=feats
    normX,means,stds=normalize(trainX, allX)
    w,b=train_logreg(normX, labels)
    probs=predict(normX,w,b)
    def subset(idxs):
        y=[labels[i] for i in idxs]
        s=[probs[i] for i in idxs]
        return {'roc_auc': auc_score(y,s), 'pr_auc': pr_auc(y,s), 'balanced_accuracy': balanced_accuracy(y,s)}
    return subset(tr), subset(va), subset(te), w

class Canvas:
    def __init__(self,w,h,bg=(255,255,255)):
        self.w=w; self.h=h; self.px=bytearray(bg*(w*h))
    def set(self,x,y,c):
        if 0<=x<self.w and 0<=y<self.h:
            i=(y*self.w+x)*3; self.px[i:i+3]=bytes(c)
    def dot(self,x,y,c,r=2):
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


def plot_dataset_overview(summary, path):
    c=Canvas(1000,520)
    x0,y0,x1,y1=70,40,950,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    names=list(summary.keys())
    vals=[summary[n]['n_samples'] for n in names]
    vmax=max(vals)
    bw=(x1-x0-40)//len(names)
    for i,n in enumerate(names):
        left=x0+20+i*bw
        right=left+bw-20
        top=y1-int(summary[n]['n_samples']/vmax*(y1-y0-40))
        c.fillrect(left,top,right,y1-1,(80,140,220))
    c.save_png(path)


def plot_comparison(results, path, metric='roc_auc'):
    c=Canvas(1000,520)
    x0,y0,x1,y1=70,40,950,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    ds=list(results.keys())
    vals=[]
    for d in ds:
        vals.append(results[d]['baseline_test'][metric] or 0.0)
        vals.append(results[d]['ka_test'][metric] or 0.0)
    vmin=min(vals+[0.4]); vmax=max(vals+[1.0])
    groupw=(x1-x0-40)//len(ds)
    for i,d in enumerate(ds):
        base=results[d]['baseline_test'][metric] or 0.0
        ka=results[d]['ka_test'][metric] or 0.0
        for j,val,col in [(0,base,(220,120,80)),(1,ka,(60,160,100))]:
            left=x0+20+i*groupw+j*(groupw//3)
            right=left+(groupw//3)-10
            top=y1-int((val-vmin)/(vmax-vmin)*(y1-y0-40))
            c.fillrect(left,top,right,y1-1,col)
    c.save_png(path)


def plot_improvements(results, path):
    c=Canvas(1000,520)
    x0,y0,x1,y1=70,40,950,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    ds=list(results.keys())
    vals=[(results[d]['ka_test']['roc_auc'] or 0.0)-(results[d]['baseline_test']['roc_auc'] or 0.0) for d in ds]
    vmax=max(max(vals),0.01); vmin=min(min(vals),-0.01)
    for i,d in enumerate(ds):
        val=vals[i]
        left=x0+40+i*((x1-x0-80)//len(ds))
        right=left+90
        zero=y0+int((vmax)/(vmax-vmin)*(y1-y0))
        top=zero-int(val/(vmax-vmin)*(y1-y0))
        c.fillrect(left,min(top,zero),right,max(top,zero),(100,100,220) if val>=0 else (220,100,100))
    c.save_png(path)


def plot_interpretability(weights_map, path):
    c=Canvas(1000,520)
    x0,y0,x1,y1=70,40,950,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    feats=weights_map
    names=[k for k,_ in feats[:12]]
    vals=[abs(v) for _,v in feats[:12]]
    vmax=max(vals) if vals else 1.0
    bw=(x1-x0-40)//max(len(vals),1)
    for i,val in enumerate(vals):
        left=x0+20+i*bw
        right=left+bw-8
        top=y1-int(val/vmax*(y1-y0-40))
        c.fillrect(left,top,right,y1-1,(140,80,180))
    c.save_png(path)


def main():
    feature_names = [f'atom_{a}' for a in ATOMS] + ['heavy','hetero','halogens','aromatic','ring_digits','branches','double_bonds','triple_bonds','charges','noncovalent_proxy'] + [f'sin_{i+1}' for i in range(8)] + [f'cos_{i+1}' for i in range(8)]
    results={}
    summary={}
    all_interpret=[]
    for fname in TASK_FILES:
        rows=parse_dataset(DATA/fname)
        features=[smiles_features(sm) for sm,_ in rows]
        splits=split_rows(rows)
        task_labels=list(zip(*[labs for _,labs in rows]))
        task_metrics=[]
        for tidx,labels in enumerate(task_labels):
            b_tr,b_va,b_te,b_w=evaluate_task(features, list(labels), splits, 'baseline')
            k_tr,k_va,k_te,k_w=evaluate_task(features, list(labels), splits, 'ka')
            task_metrics.append({
                'task_index': tidx,
                'baseline_test': b_te,
                'ka_test': k_te,
                'baseline_val': b_va,
                'ka_val': k_va,
                'weight_vector': k_w,
            })
        baseline_test={m: avg([tm['baseline_test'][m] for tm in task_metrics]) for m in ['roc_auc','pr_auc','balanced_accuracy']}
        ka_test={m: avg([tm['ka_test'][m] for tm in task_metrics]) for m in ['roc_auc','pr_auc','balanced_accuracy']}
        baseline_val={m: avg([tm['baseline_val'][m] for tm in task_metrics]) for m in ['roc_auc','pr_auc','balanced_accuracy']}
        ka_val={m: avg([tm['ka_val'][m] for tm in task_metrics]) for m in ['roc_auc','pr_auc','balanced_accuracy']}
        label_counts=[]
        for labels in task_labels:
            pos=sum(1 for x in labels if x==1); neg=sum(1 for x in labels if x==0); miss=sum(1 for x in labels if x is None)
            label_counts.append({'positive':pos,'negative':neg,'missing':miss})
        results[fname.replace('.csv','')]={'baseline_test':baseline_test,'ka_test':ka_test,'baseline_val':baseline_val,'ka_val':ka_val,'tasks':task_metrics}
        summary[fname.replace('.csv','')]={'n_samples':len(rows),'n_tasks':len(task_labels),'label_counts':label_counts}
        if task_metrics:
            w=task_metrics[0]['weight_vector']
            all_interpret.extend(list(zip(feature_names,w)))
    mean_weights={}
    for name,val in all_interpret:
        mean_weights.setdefault(name,[]).append(val)
    interpret=sorted([(k,sum(v)/len(v)) for k,v in mean_weights.items()], key=lambda x: abs(x[1]), reverse=True)
    (OUT/'results.json').write_text(json.dumps({'summary':summary,'results':results,'interpretability':interpret}, indent=2), encoding='utf-8')
    with open(OUT/'results_table.csv','w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['dataset','baseline_roc_auc','ka_roc_auc','baseline_pr_auc','ka_pr_auc','baseline_bal_acc','ka_bal_acc'])
        for ds,res in results.items():
            w.writerow([ds,res['baseline_test']['roc_auc'],res['ka_test']['roc_auc'],res['baseline_test']['pr_auc'],res['ka_test']['pr_auc'],res['baseline_test']['balanced_accuracy'],res['ka_test']['balanced_accuracy']])
    plot_dataset_overview(summary, IMG/'data_overview.png')
    plot_comparison(results, IMG/'main_results.png', 'roc_auc')
    plot_comparison(results, IMG/'validation_comparison.png', 'pr_auc')
    plot_improvements(results, IMG/'improvement_plot.png')
    plot_interpretability(interpret, IMG/'interpretability.png')
    print(json.dumps({'summary':summary, 'table':{k:{'baseline_test':v['baseline_test'],'ka_test':v['ka_test']} for k,v in results.items()}, 'interpretability_top':interpret[:10]}, indent=2))

if __name__=='__main__':
    main()
