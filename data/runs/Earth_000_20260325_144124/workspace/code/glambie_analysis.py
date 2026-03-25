import csv, json, math, zlib, struct
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / 'data' / 'glambie'
OUT = BASE / 'outputs'
IMG = BASE / 'report' / 'images'
OUT.mkdir(exist_ok=True)
IMG.mkdir(parents=True, exist_ok=True)

REGION_NAMES = {
    '1_alaska':'Alaska', '2_western_canada_us':'Western Canada & US', '3_arctic_canada_north':'Arctic Canada North',
    '4_arctic_canada_south':'Arctic Canada South', '5_greenland_periphery':'Greenland Periphery', '6_iceland':'Iceland',
    '7_svalbard':'Svalbard', '8_scandinavia':'Scandinavia', '9_russian_arctic':'Russian Arctic', '10_north_asia':'North Asia',
    '11_central_europe':'Central Europe', '12_caucasus_middle_east':'Caucasus & Middle East', '13_central_asia':'Central Asia',
    '14_south_asia_west':'South Asia West', '15_south_asia_east':'South Asia East', '16_low_latitudes':'Low Latitudes',
    '17_southern_andes':'Southern Andes', '18_new_zealand':'New Zealand', '19_antarctic_and_subantarctic':'Antarctic & Subantarctic',
    '0_global':'Global'
}

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


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def to_float(x):
    try:
        if x is None or x == '': return None
        return float(x)
    except:
        return None


def load_calendar_results():
    res={}
    for p in sorted((DATA/'results'/'calendar_years').glob('*.csv')):
        key=p.stem
        rows=[]
        for r in read_csv(p):
            rows.append({k: to_float(v) if k not in ('region',) else v for k,v in r.items()})
        res[key]=rows
    return res


def load_hydro_results():
    res={}
    for p in sorted((DATA/'results'/'hydrological_years').glob('*.csv')):
        key=p.stem
        rows=[]
        for r in read_csv(p):
            rows.append({k: to_float(v) if k not in ('region',) else v for k,v in r.items()})
        res[key]=rows
    return res


def input_inventory():
    region_files={}
    for p in sorted((DATA/'input').glob('*/*.csv')):
        region=p.parent.name
        region_files.setdefault(region, []).append(p.name)
    return region_files


def weighted_mean(vals, errs):
    pairs=[(v,e) for v,e in zip(vals,errs) if v is not None and e is not None and e>0]
    if not pairs: return None, None
    ws=[1.0/(e*e) for _,e in pairs]
    mu=sum(v*w for (v,_),w in zip(pairs,ws))/sum(ws)
    se=math.sqrt(1.0/sum(ws))
    return mu, se


def summarize(calendar, hydro, inventory):
    global_rows=calendar['0_global']
    total_gt=sum(r['combined_gt'] for r in global_rows)
    total_gt_err=math.sqrt(sum((r['combined_gt_errors'] or 0.0)**2 for r in global_rows))
    total_mwe=sum(r['combined_mwe'] for r in global_rows)
    mean_annual_gt=total_gt/len(global_rows)
    mean_annual_mwe=total_mwe/len(global_rows)
    peak_loss=min(global_rows, key=lambda r:r['combined_gt'])
    least_loss=max(global_rows, key=lambda r:r['combined_gt'])
    region_totals=[]
    for k,rows in calendar.items():
        if k=='0_global': continue
        region_totals.append({
            'region_key':k,
            'region':REGION_NAMES.get(k,k),
            'total_gt':sum(r['combined_gt'] for r in rows),
            'mean_annual_gt':sum(r['combined_gt'] for r in rows)/len(rows),
            'mean_annual_mwe':sum(r['combined_mwe'] for r in rows)/len(rows),
            'last5_mean_gt':sum(r['combined_gt'] for r in rows[-5:])/5.0,
            'n_years':len(rows),
            'area_2000':rows[0]['glacier_area'],
            'area_2023':rows[-1]['glacier_area'],
            'input_estimates':len(inventory.get(k,[])),
        })
    region_totals.sort(key=lambda x:x['total_gt'])
    method_stats=[]
    for k,rows in hydro.items():
        if k=='0_global':
            continue
        fields=['altimetry_gt','gravimetry_gt','demdiff_and_glaciological_gt']
        for fld in fields:
            vals=[r.get(fld) for r in rows]
            errs=[r.get(fld+'_errors') for r in rows]
            avail=sum(1 for v in vals if v is not None)
            mu,se=weighted_mean(vals,errs)
            method_stats.append({'region_key':k,'method':fld.replace('_gt',''),'available_years':avail,'weighted_mean_gt':mu,'weighted_mean_gt_error':se})
    return {
        'global':{
            'n_years':len(global_rows),
            'total_gt_2000_2023':total_gt,
            'cumulative_error_gt_quadrature':total_gt_err,
            'mean_annual_gt':mean_annual_gt,
            'total_mwe_2000_2023':total_mwe,
            'mean_annual_mwe':mean_annual_mwe,
            'peak_loss_year_start':peak_loss['start_dates'],
            'peak_loss_gt':peak_loss['combined_gt'],
            'least_loss_year_start':least_loss['start_dates'],
            'least_loss_gt':least_loss['combined_gt'],
            'area_2000':global_rows[0]['glacier_area'],
            'area_2023':global_rows[-1]['glacier_area'],
        },
        'regions_ranked_by_total_gt':region_totals,
        'method_stats_sample':method_stats,
        'inventory_counts':{k:len(v) for k,v in inventory.items()},
    }


def plot_global_series(calendar, path):
    rows=calendar['0_global']
    years=[r['start_dates'] for r in rows]
    vals=[r['combined_gt'] for r in rows]
    errs=[r['combined_gt_errors'] for r in rows]
    c=Canvas(1000,520)
    x0,y0,x1,y1=70,40,950,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    ymin=min(v-e for v,e in zip(vals,errs)); ymax=max(v+e for v,e in zip(vals,errs))
    ymin=min(ymin,-10); ymax=max(ymax,10)
    def xy(x,y):
        px=x0+int((x-years[0])/(years[-1]-years[0])*(x1-x0))
        py=y1-int((y-ymin)/(ymax-ymin)*(y1-y0))
        return px,py
    zero1=xy(years[0],0); zero2=xy(years[-1],0)
    c.line(zero1[0],zero1[1],zero2[0],zero2[1],(160,160,160),1)
    pts=[xy(x,y) for x,y in zip(years,vals)]
    for (x,y),(xe,ve,er) in zip(pts, zip(years,vals,errs)):
        _,py1=xy(xe, ve-er); _,py2=xy(xe, ve+er)
        c.line(x,py1,x,py2,(180,180,180),1)
    for p,q in zip(pts[:-1],pts[1:]): c.line(p[0],p[1],q[0],q[1],(30,90,170),2)
    c.save_png(path)


def plot_regional_bars(summary, path):
    regs=summary['regions_ranked_by_total_gt'][:10]
    c=Canvas(1000,520)
    x0,y0,x1,y1=70,40,950,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    vals=[r['total_gt'] for r in regs]
    vmin=min(vals); vmax=max(vals+[0.0])
    bw=(x1-x0-40)//len(regs)
    zero=y0+int((vmax)/(vmax-vmin)*(y1-y0)) if vmax!=vmin else (y0+y1)//2
    for i,r in enumerate(regs):
        val=r['total_gt']
        left=x0+20+i*bw; right=left+bw-10
        top=zero-int((val)/(vmax-vmin)*(y1-y0)) if vmax!=vmin else zero
        c.fillrect(left,min(top,zero),right,max(top,zero),(200,90,90))
    c.save_png(path)


def plot_method_availability(summary, path):
    counts={m:0 for m in ['altimetry','gravimetry','demdiff_and_glaciological']}
    yearsums={m:0 for m in counts}
    for rec in summary['method_stats_sample']:
        if rec['available_years']>0:
            counts[rec['method']] += 1
            yearsums[rec['method']] += rec['available_years']
    c=Canvas(900,520)
    x0,y0,x1,y1=70,40,850,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    methods=list(counts.keys())
    vals=[yearsums[m] for m in methods]
    vmax=max(vals) if vals else 1
    bw=(x1-x0-40)//len(methods)
    cols=[(80,140,220),(120,180,100),(180,120,220)]
    for i,m in enumerate(methods):
        left=x0+20+i*bw; right=left+bw-30
        top=y1-int(vals[i]/vmax*(y1-y0-30))
        c.fillrect(left,top,right,y1-1,cols[i])
    c.save_png(path)


def plot_validation(calendar, hydro, path):
    # compare calendar vs hydrological global annual totals using overlapping years by start year
    cal={int(r['start_dates']):r['combined_gt'] for r in calendar['0_global']}
    # synthesize hydro global from regional sums because no hydro global file
    hydro_years={}
    for k,rows in hydro.items():
        if k=='0_global': continue
        for r in rows:
            y=int(math.floor(r['start_dates']))
            hydro_years.setdefault(y,0.0)
            hydro_years[y]+=r['combined_gt']
    years=sorted(y for y in cal if y in hydro_years)
    c=Canvas(1000,520)
    x0,y0,x1,y1=70,40,950,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    vals=[cal[y] for y in years]+[hydro_years[y] for y in years]
    ymin=min(vals); ymax=max(vals)
    def xy(x,y):
        px=x0+int((x-years[0])/(years[-1]-years[0])*(x1-x0)) if years[-1]>years[0] else (x0+x1)//2
        py=y1-int((y-ymin)/(ymax-ymin)*(y1-y0)) if ymax>ymin else (y0+y1)//2
        return px,py
    pts1=[xy(y,cal[y]) for y in years]; pts2=[xy(y,hydro_years[y]) for y in years]
    for p,q in zip(pts1[:-1],pts1[1:]): c.line(p[0],p[1],q[0],q[1],(30,90,170),2)
    for p,q in zip(pts2[:-1],pts2[1:]): c.line(p[0],p[1],q[0],q[1],(220,120,40),2)
    c.save_png(path)


def main():
    calendar=load_calendar_results()
    hydro=load_hydro_results()
    inventory=input_inventory()
    summary=summarize(calendar, hydro, inventory)
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    with open(OUT/'global_series.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['start_date','end_date','combined_gt','combined_gt_errors','combined_mwe','combined_mwe_errors'])
        for r in calendar['0_global']:
            w.writerow([r['start_dates'],r['end_dates'],r['combined_gt'],r['combined_gt_errors'],r['combined_mwe'],r['combined_mwe_errors']])
    with open(OUT/'regional_ranking.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f)
        w.writerow(['region','total_gt','mean_annual_gt','mean_annual_mwe','last5_mean_gt','n_years','input_estimates'])
        for r in summary['regions_ranked_by_total_gt']:
            w.writerow([r['region'],r['total_gt'],r['mean_annual_gt'],r['mean_annual_mwe'],r['last5_mean_gt'],r['n_years'],r['input_estimates']])
    plot_global_series(calendar, IMG/'global_time_series.png')
    plot_regional_bars(summary, IMG/'regional_contributions.png')
    plot_method_availability(summary, IMG/'method_availability.png')
    plot_validation(calendar, hydro, IMG/'validation_comparison.png')
    # overview figure: counts of regional estimates per region from input inventory
    c=Canvas(1000,520)
    x0,y0,x1,y1=70,40,950,470
    c.rect(x0,y0,x1,y1,(0,0,0))
    regs=sorted((k,v) for k,v in summary['inventory_counts'].items())
    vals=[v for _,v in regs]; vmax=max(vals)
    bw=max(6,(x1-x0-30)//len(regs))
    for i,(k,v) in enumerate(regs):
        left=x0+15+i*bw; right=min(left+bw-2,x1-2)
        top=y1-int(v/vmax*(y1-y0-20))
        c.fillrect(left,top,right,y1-1,(90,150,220))
    c.save_png(IMG/'data_overview.png')
    print(json.dumps(summary['global'], indent=2))
    print('top loss regions', [r['region'] for r in summary['regions_ranked_by_total_gt'][:5]])

if __name__=='__main__':
    main()
