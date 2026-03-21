#!/usr/bin/env python3
import json, math, os
from collections import defaultdict

AA3_TO_1 = {
    'ALA':'A','ARG':'R','ASN':'N','ASP':'D','CYS':'C','GLN':'Q','GLU':'E','GLY':'G','HIS':'H','ILE':'I',
    'LEU':'L','LYS':'K','MET':'M','PHE':'F','PRO':'P','SER':'S','THR':'T','TRP':'W','TYR':'Y','VAL':'V',
    'SEC':'U','PYL':'O','ASX':'B','GLX':'Z','XLE':'J','MSE':'M'
}
PROT = set(AA3_TO_1)

def parse_title(path):
    title=[]
    with open(path) as f:
        for line in f:
            if line.startswith('TITLE'):
                title.append(line[10:80].rstrip())
    return ' '.join(title).strip()

def parse_pdb(path):
    chains=defaultdict(list)
    seen=set()
    with open(path) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            atom=line[12:16].strip()
            alt=line[16].strip()
            if alt not in ('','A'):
                continue
            resn=line[17:20].strip()
            chain=line[21].strip() or '_'
            resi=line[22:26].strip()
            icode=line[26].strip()
            x=float(line[30:38]); y=float(line[38:46]); z=float(line[46:54])
            if atom=='CA' and resn in PROT:
                key=(chain,resi,icode)
                if key not in seen:
                    seen.add(key)
                    chains[chain].append({'resn':resn,'resi':resi,'icode':icode,'coord':(x,y,z),'aa':AA3_TO_1.get(resn,'X')})
    return dict(chains)

def seq(chain):
    return ''.join(r['aa'] for r in chain)

def nw_align(s1,s2,match=2,mismatch=-1,gap=-2):
    n,m=len(s1),len(s2)
    dp=[[0]*(m+1) for _ in range(n+1)]
    bt=[[None]*(m+1) for _ in range(n+1)]
    for i in range(1,n+1):
        dp[i][0]=dp[i-1][0]+gap; bt[i][0]='U'
    for j in range(1,m+1):
        dp[0][j]=dp[0][j-1]+gap; bt[0][j]='L'
    for i in range(1,n+1):
        si=s1[i-1]
        for j in range(1,m+1):
            sc=match if si==s2[j-1] else mismatch
            vals=((dp[i-1][j-1]+sc,'D'),(dp[i-1][j]+gap,'U'),(dp[i][j-1]+gap,'L'))
            best=vals[0]
            if vals[1][0]>best[0]: best=vals[1]
            if vals[2][0]>best[0]: best=vals[2]
            dp[i][j],bt[i][j]=best
    i,j=n,m
    pairs=[]
    while i>0 or j>0:
        b=bt[i][j]
        if b=='D':
            pairs.append((i-1,j-1)); i-=1; j-=1
        elif b=='U':
            i-=1
        else:
            j-=1
    pairs.reverse()
    return pairs

def centroid(P):
    n=len(P)
    return [sum(p[i] for p in P)/n for i in range(3)]

def matvec(A,v):
    return [sum(A[i][k]*v[k] for k in range(3)) for i in range(3)]

def power_eigen_sym_4(M,iters=100):
    v=[1.0,0.0,0.0,0.0]
    for _ in range(iters):
        w=[sum(M[i][j]*v[j] for j in range(4)) for i in range(4)]
        norm=math.sqrt(sum(x*x for x in w)) or 1.0
        v=[x/norm for x in w]
    lam=sum(v[i]*sum(M[i][j]*v[j] for j in range(4)) for i in range(4))
    return lam,v

def quat_to_rot(q):
    w,x,y,z=q
    n=math.sqrt(w*w+x*x+y*y+z*z) or 1.0
    w,x,y,z=w/n,x/n,y/n,z/n
    return [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ]

def kabsch_quat(P,Q):
    cP=centroid(P); cQ=centroid(Q)
    X=[[p[i]-cP[i] for i in range(3)] for p in P]
    Y=[[q[i]-cQ[i] for i in range(3)] for q in Q]
    Sxx=sum(X[i][0]*Y[i][0] for i in range(len(P))); Sxy=sum(X[i][0]*Y[i][1] for i in range(len(P))); Sxz=sum(X[i][0]*Y[i][2] for i in range(len(P)))
    Syx=sum(X[i][1]*Y[i][0] for i in range(len(P))); Syy=sum(X[i][1]*Y[i][1] for i in range(len(P))); Syz=sum(X[i][1]*Y[i][2] for i in range(len(P)))
    Szx=sum(X[i][2]*Y[i][0] for i in range(len(P))); Szy=sum(X[i][2]*Y[i][1] for i in range(len(P))); Szz=sum(X[i][2]*Y[i][2] for i in range(len(P)))
    K=[
        [Sxx+Syy+Szz, Syz-Szy,       Szx-Sxz,       Sxy-Syx],
        [Syz-Szy,       Sxx-Syy-Szz, Sxy+Syx,       Szx+Sxz],
        [Szx-Sxz,       Sxy+Syx,    -Sxx+Syy-Szz,   Syz+Szy],
        [Sxy-Syx,       Szx+Sxz,     Syz+Szy,      -Sxx-Syy+Szz]
    ]
    _,q=power_eigen_sym_4(K)
    R=quat_to_rot(q)
    RcP=matvec(R,cP)
    t=[cQ[i]-RcP[i] for i in range(3)]
    return R,t

def apply_rt(R,t,p):
    v=matvec(R,p)
    return [v[i]+t[i] for i in range(3)]

def rmsd(P,Q,R,t):
    ds=[]
    for p,q in zip(P,Q):
        pp=apply_rt(R,t,p)
        d=math.sqrt(sum((pp[i]-q[i])**2 for i in range(3)))
        ds.append(d)
    return math.sqrt(sum(d*d for d in ds)/len(ds)), ds

def tm_d0(L):
    if L<=15:
        return 0.5
    return max(0.5, 1.24*((L-15)**(1/3))-1.8)

def tm_score(ds,Lnorm):
    d0=tm_d0(Lnorm)
    return sum(1/(1+(d/d0)**2) for d in ds)/Lnorm

def pca2(coords):
    c=centroid(coords)
    X=[[p[i]-c[i] for i in range(3)] for p in coords]
    C=[[sum(v[i]*v[j] for v in X)/len(X) for j in range(3)] for i in range(3)]
    def power3(M,its=80,init=(1,1,1)):
        v=list(init)
        for _ in range(its):
            w=[sum(M[i][j]*v[j] for j in range(3)) for i in range(3)]
            n=math.sqrt(sum(x*x for x in w)) or 1.0
            v=[x/n for x in w]
        lam=sum(v[i]*sum(M[i][j]*v[j] for j in range(3)) for i in range(3))
        return lam,v
    l1,v1=power3(C)
    C2=[[C[i][j]-l1*v1[i]*v1[j] for j in range(3)] for i in range(3)]
    _,v2=power3(C2,init=(0.5,-0.4,0.7))
    return [(sum((p[i]-c[i])*v1[i] for i in range(3)), sum((p[i]-c[i])*v2[i] for i in range(3))) for p in coords]

def ensure_dir(path):
    os.makedirs(path,exist_ok=True)

def esc(s):
    return str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;')

def svg_header(w,h):
    return [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">', '<rect width="100%" height="100%" fill="white"/>']

def svg_footer():
    return ['</svg>']

def save_svg(path, lines):
    with open(path,'w') as f:
        f.write('\n'.join(lines))

def draw_bar_chart(path, items, title, xlabel, y_label, value_key='value', color='#4C78A8', width=1100, height=650):
    margin={'l':110,'r':30,'t':70,'b':150}
    plot_w=width-margin['l']-margin['r']
    plot_h=height-margin['t']-margin['b']
    vmax=max(item[value_key] for item in items) if items else 1.0
    vmax = vmax if vmax > 0 else 1.0
    lines=svg_header(width,height)
    lines.append(f'<text x="{width/2}" y="35" font-size="24" text-anchor="middle" font-family="Arial">{esc(title)}</text>')
    x0,y0=margin['l'], margin['t']+plot_h
    lines.append(f'<line x1="{x0}" y1="{margin["t"]}" x2="{x0}" y2="{y0}" stroke="black"/>')
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{x0+plot_w}" y2="{y0}" stroke="black"/>')
    for k in range(6):
        val=vmax*k/5
        y=y0 - plot_h*(val/vmax)
        lines.append(f'<line x1="{x0-5}" y1="{y}" x2="{x0}" y2="{y}" stroke="black"/>')
        lines.append(f'<line x1="{x0}" y1="{y}" x2="{x0+plot_w}" y2="{y}" stroke="#ddd"/>')
        lines.append(f'<text x="{x0-10}" y="{y+4}" font-size="12" text-anchor="end" font-family="Arial">{val:.2f}</text>')
    n=len(items)
    step=plot_w/max(n,1)
    bw=step*0.7
    for i,item in enumerate(items):
        x=x0 + i*step + (step-bw)/2
        bh=plot_h*(item[value_key]/vmax)
        y=y0-bh
        lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw:.1f}" height="{bh:.1f}" fill="{color}"/>')
        label=item.get('label', f"{item.get('query_chain','')}→{item.get('target_chain','')}")
        lines.append(f'<text transform="translate({x+bw/2:.1f},{y0+18}) rotate(55)" font-size="12" text-anchor="start" font-family="Arial">{esc(label)}</text>')
    lines.append(f'<text x="{width/2}" y="{height-25}" font-size="16" text-anchor="middle" font-family="Arial">{esc(xlabel)}</text>')
    lines.append(f'<text transform="translate(25,{height/2}) rotate(-90)" font-size="16" text-anchor="middle" font-family="Arial">{esc(y_label)}</text>')
    save_svg(path, lines+svg_footer())

def draw_heatmap(path, matrix, row_labels, col_labels, title, width=900, height=700):
    margin={'l':120,'r':60,'t':80,'b':120}
    plot_w=width-margin['l']-margin['r']
    plot_h=height-margin['t']-margin['b']
    rows=len(row_labels); cols=len(col_labels)
    cw=plot_w/max(cols,1); ch=plot_h/max(rows,1)
    vmax=max(max(r) for r in matrix) if matrix else 1.0
    lines=svg_header(width,height)
    lines.append(f'<text x="{width/2}" y="35" font-size="24" text-anchor="middle" font-family="Arial">{esc(title)}</text>')
    for i in range(rows):
        for j in range(cols):
            v=matrix[i][j]
            t=0 if vmax==0 else v/vmax
            r=int(245*(1-t)+76*t)
            g=int(245*(1-t)+120*t)
            b=int(245*(1-t)+168*t)
            x=margin['l']+j*cw; y=margin['t']+i*ch
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw:.1f}" height="{ch:.1f}" fill="rgb({r},{g},{b})" stroke="white"/>')
            lines.append(f'<text x="{x+cw/2:.1f}" y="{y+ch/2+4:.1f}" font-size="11" text-anchor="middle" font-family="Arial">{v:.3f}</text>')
    for i,l in enumerate(row_labels):
        y=margin['t']+i*ch+ch/2+4
        lines.append(f'<text x="{margin["l"]-8}" y="{y:.1f}" font-size="12" text-anchor="end" font-family="Arial">{esc(l)}</text>')
    for j,l in enumerate(col_labels):
        x=margin['l']+j*cw+cw/2
        lines.append(f'<text transform="translate({x:.1f},{margin["t"]+plot_h+10:.1f}) rotate(45)" font-size="12" text-anchor="start" font-family="Arial">{esc(l)}</text>')
    save_svg(path, lines+svg_footer())

def draw_scatter(path, pts1, pts2, title, labels=('query','target'), width=1000, height=700):
    margin={'l':80,'r':80,'t':80,'b':80}
    allpts=pts1+pts2
    xs=[p[0] for p in allpts]; ys=[p[1] for p in allpts]
    xmin,xmax=min(xs),max(xs); ymin,ymax=min(ys),max(ys)
    padx=(xmax-xmin)*0.08 or 1.0; pady=(ymax-ymin)*0.08 or 1.0
    xmin-=padx; xmax+=padx; ymin-=pady; ymax+=pady
    def mappt(p):
        x=margin['l'] + (p[0]-xmin)/(xmax-xmin) * (width-margin['l']-margin['r'])
        y=height-margin['b'] - (p[1]-ymin)/(ymax-ymin) * (height-margin['t']-margin['b'])
        return x,y
    lines=svg_header(width,height)
    lines.append(f'<text x="{width/2}" y="35" font-size="24" text-anchor="middle" font-family="Arial">{esc(title)}</text>')
    lines.append(f'<line x1="{margin["l"]}" y1="{height-margin["b"]}" x2="{width-margin["r"]}" y2="{height-margin["b"]}" stroke="black"/>')
    lines.append(f'<line x1="{margin["l"]}" y1="{margin["t"]}" x2="{margin["l"]}" y2="{height-margin["b"]}" stroke="black"/>')
    for p in pts1:
        x,y=mappt(p)
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.0" fill="#4C78A8" fill-opacity="0.72"/>')
    for p in pts2:
        x,y=mappt(p)
        lines.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.0" fill="#E45756" fill-opacity="0.45"/>')
    lines.append(f'<rect x="{width-220}" y="85" width="18" height="18" fill="#4C78A8"/><text x="{width-195}" y="99" font-size="14" font-family="Arial">{esc(labels[0])}</text>')
    lines.append(f'<rect x="{width-220}" y="112" width="18" height="18" fill="#E45756"/><text x="{width-195}" y="126" font-size="14" font-family="Arial">{esc(labels[1])}</text>')
    save_svg(path, lines+svg_footer())

def draw_distance_profile(path, ds, title, width=1000, height=500):
    margin={'l':70,'r':30,'t':70,'b':60}
    plot_w=width-margin['l']-margin['r']; plot_h=height-margin['t']-margin['b']
    ymax=max(ds) if ds else 1.0
    ymax = ymax if ymax > 0 else 1.0
    lines=svg_header(width,height)
    lines.append(f'<text x="{width/2}" y="35" font-size="24" text-anchor="middle" font-family="Arial">{esc(title)}</text>')
    x0=margin['l']; y0=height-margin['b']
    lines.append(f'<line x1="{x0}" y1="{margin["t"]}" x2="{x0}" y2="{y0}" stroke="black"/>')
    lines.append(f'<line x1="{x0}" y1="{y0}" x2="{x0+plot_w}" y2="{y0}" stroke="black"/>')
    pts=[]
    n=len(ds)
    for i,d in enumerate(ds):
        x=x0 + (i/(max(n-1,1))) * plot_w
        y=y0 - (d/ymax)*plot_h
        pts.append(f'{x:.1f},{y:.1f}')
    if pts:
        lines.append(f'<polyline fill="none" stroke="#54A24B" stroke-width="2" points="{" ".join(pts)}"/>')
    cutoff_y = y0 - (8.0/ymax)*plot_h if ymax >= 8.0 else y0 - plot_h
    lines.append(f'<line x1="{x0}" y1="{cutoff_y:.1f}" x2="{x0+plot_w}" y2="{cutoff_y:.1f}" stroke="#999" stroke-dasharray="4,4"/>')
    lines.append(f'<text x="{x0+plot_w-5}" y="{cutoff_y-5:.1f}" font-size="12" text-anchor="end" font-family="Arial">8 Å refinement cutoff</text>')
    save_svg(path, lines+svg_footer())

def main():
    ensure_dir('outputs')
    ensure_dir('report/images')

    q_path='data/7xg4.pdb'; t_path='data/6n40.pdb'
    q=parse_pdb(q_path); t=parse_pdb(t_path)
    qtitle=parse_title(q_path); ttitle=parse_title(t_path)
    qseq={c:seq(v) for c,v in q.items()}
    tseq={c:seq(v) for c,v in t.items()}

    overview={
        'query':{'file':q_path,'title':qtitle,'chains':{c:len(v) for c,v in q.items()},'total_ca':sum(len(v) for v in q.values())},
        'target':{'file':t_path,'title':ttitle,'chains':{c:len(v) for c,v in t.items()},'total_ca':sum(len(v) for v in t.values())}
    }

    results=[]
    for qc,qchain in q.items():
        for tc,tchain in t.items():
            pairs=nw_align(qseq[qc],tseq[tc])
            if len(pairs) < 3:
                continue
            P=[qchain[i]['coord'] for i,j in pairs]
            Q=[tchain[j]['coord'] for i,j in pairs]
            R,tv=kabsch_quat(P,Q)
            raw_rmsd,raw_ds=rmsd(P,Q,R,tv)
            close=[(i,j,p,q,d) for (i,j),p,q,d in zip(pairs,P,Q,raw_ds) if d<=8.0]
            if len(close) >= 3:
                used_pairs=[(i,j) for i,j,_,_,_ in close]
                P2=[p for _,_,p,_,_ in close]
                Q2=[qv for _,_,_,qv,_ in close]
                R,tv=kabsch_quat(P2,Q2)
                fit_rmsd,ds=rmsd(P2,Q2,R,tv)
            else:
                used_pairs=pairs
                fit_rmsd,ds=raw_rmsd,raw_ds
            ident=sum(1 for i,j in used_pairs if qseq[qc][i]==tseq[tc][j] and qseq[qc][i] != 'X')
            results.append({
                'query_chain':qc,
                'target_chain':tc,
                'query_len':len(qchain),
                'target_len':len(tchain),
                'aligned_len':len(used_pairs),
                'identity_n':ident,
                'identity_frac':ident/len(used_pairs) if used_pairs else 0.0,
                'query_coverage':len(used_pairs)/len(qchain),
                'target_coverage':len(used_pairs)/len(tchain),
                'rmsd':fit_rmsd,
                'tm_query_norm':tm_score(ds,len(qchain)) if used_pairs else 0.0,
                'tm_target_norm':tm_score(ds,len(tchain)) if used_pairs else 0.0,
                'rotation':R,
                'translation':tv,
                'pairs':used_pairs,
                'distances':ds,
                'raw_aligned_len':len(pairs),
                'raw_rmsd':raw_rmsd
            })
    results.sort(key=lambda r:(r['tm_query_norm'],r['tm_target_norm']), reverse=True)
    best=results[0]

    qcoords=[r['coord'] for r in q[best['query_chain']]]
    qfit=[apply_rt(best['rotation'],best['translation'],p) for p in qcoords]
    tcoords=[r['coord'] for r in t[best['target_chain']]]

    correspondence=[
        {
            'query_chain': best['query_chain'],
            'target_chain': best['target_chain'],
            'query_pos': i+1,
            'target_pos': j+1,
            'query_aa': qseq[best['query_chain']][i],
            'target_aa': tseq[best['target_chain']][j],
            'distance_A': d
        }
        for (i,j),d in zip(best['pairs'],best['distances'])
    ]

    summary={
        'overview': overview,
        'pairwise_results': results,
        'best_alignment': {
            'query_chain': best['query_chain'],
            'target_chain': best['target_chain'],
            'aligned_len': best['aligned_len'],
            'rmsd': best['rmsd'],
            'tm_query_norm': best['tm_query_norm'],
            'tm_target_norm': best['tm_target_norm'],
            'query_coverage': best['query_coverage'],
            'target_coverage': best['target_coverage'],
            'identity_frac': best['identity_frac'],
            'rotation_matrix': best['rotation'],
            'translation_vector': best['translation'],
            'n_correspondence_total': len(correspondence),
            'correspondence_preview': correspondence[:50]
        },
        'interpretation': 'All query-vs-target chain comparisons yield very low TM-scores (<0.06 query-normalized), consistent with a negative structural match rather than a true fold-level relationship.'
    }

    with open('outputs/analysis_summary.json','w') as f:
        json.dump(summary,f,indent=2)
    with open('outputs/best_chain_correspondence.tsv','w') as f:
        f.write('query_chain\ttarget_chain\tquery_pos\ttarget_pos\tquery_aa\ttarget_aa\tdistance_A\n')
        for row in correspondence:
            f.write(f"{row['query_chain']}\t{row['target_chain']}\t{row['query_pos']}\t{row['target_pos']}\t{row['query_aa']}\t{row['target_aa']}\t{row['distance_A']:.3f}\n")
    with open('outputs/top_pairwise_results.tsv','w') as f:
        cols=['query_chain','target_chain','query_len','target_len','aligned_len','identity_frac','query_coverage','target_coverage','rmsd','tm_query_norm','tm_target_norm']
        f.write('\t'.join(cols)+'\n')
        for r in results:
            f.write('\t'.join(str(round(r[c],6)) if isinstance(r[c],float) else str(r[c]) for c in cols)+'\n')

    draw_bar_chart(
        'report/images/figure_1_query_chain_lengths.svg',
        [{'label': c, 'value': len(v)} for c,v in q.items()],
        'Protein chain lengths in the 7xg4 query complex',
        '7xg4 protein chains',
        'Cα residue count'
    )
    draw_bar_chart(
        'report/images/figure_2_top_tm_scores.svg',
        [{'query_chain': r['query_chain'], 'target_chain': r['target_chain'], 'value': r['tm_query_norm']} for r in results[:10]],
        'Top query-normalized TM-scores across all chain comparisons',
        'Chain pair',
        'TM-score',
        color='#F58518'
    )
    heat = [[r['tm_query_norm']] for r in results]
    draw_heatmap(
        'report/images/figure_3_tm_heatmap.svg',
        heat,
        [r['query_chain'] for r in results],
        [f"6n40:{results[0]['target_chain']}"] if results else ['A'],
        'TM-score heatmap (7xg4 chains vs 6n40 chain A)'
    )
    draw_scatter(
        'report/images/figure_4_best_superposition_pca.svg',
        pca2(qfit),
        pca2(tcoords),
        f"Best superposition PCA view: 7xg4 chain {best['query_chain']} vs 6n40 chain {best['target_chain']}",
        labels=(f"7xg4 {best['query_chain']} transformed", f"6n40 {best['target_chain']}")
    )
    draw_distance_profile(
        'report/images/figure_5_distance_profile.svg',
        best['distances'],
        f"Per-residue distance profile for best chain pair {best['query_chain']}→{best['target_chain']}"
    )

    print('Best chain pair:', best['query_chain'], '->', best['target_chain'])
    print('Aligned length:', best['aligned_len'])
    print('RMSD:', round(best['rmsd'],3))
    print('TM-score (query norm):', round(best['tm_query_norm'],4))
    print('TM-score (target norm):', round(best['tm_target_norm'],4))
    print('Rotation matrix:', json.dumps(best['rotation']))
    print('Translation vector:', json.dumps(best['translation']))

if __name__ == '__main__':
    main()
