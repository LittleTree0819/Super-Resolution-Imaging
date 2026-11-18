"""Independent Huygens method/pupil/window/pixel convergence experiments.

All changes are confined to a standalone in-memory optical system. Outputs are
relative floating intensity, not PNGs or absolute radiometry. No dataset writes.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from super_resolution_imaging.config import load_config
from super_resolution_imaging.zos.connection import connect
from super_resolution_imaging.zos.model import set_lens_shift,validate_source
from super_resolution_imaging.zos.image_simulation import _sample_size

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--suite',choices=['method','pupil','window','pixel','validation','nodes','confirm','pixel_patch','high_reference'],default='method')
parser.add_argument('--pupil',type=int,default=1024)
parser.add_argument('--node-pupils',type=int,nargs='+',default=[2048,4096,8192])
args=parser.parse_args()
OUT=ROOT/'runs/psf_reliability_v2';OUT.mkdir(parents=True,exist_ok=True)
cfg=load_config(ROOT/'configs/zemax_quarter_four_frozen.toml');source=validate_source(cfg)
# tuple: mode,pupil,image grid,delta um,position,field x/y,wavelength
def job(mode,p,n,d,pos='a',x=0.,y=0.,w=.535):return (mode,p,n,d,pos,x,y,w)
if args.suite=='method':
    jobs=[job(m,512,n,1.2) for m in ('Auto','Planar','Spherical') for n in (128,256)]
elif args.suite=='pupil':
    jobs=[job('Spherical',p,128,1.2) for p in (256,512,1024,2048)]
elif args.suite=='window':
    jobs=[job('Spherical',args.pupil,n,1.2) for n in (128,256,512)]
elif args.suite=='pixel':
    jobs=[job('Spherical',args.pupil,n,d) for n,d in ((128,1.2),(256,.6),(512,.3))]
elif args.suite=='nodes':
    # Sparse intensity probes are NOT quadrature grids or production kernels.
    # They independently test coherent pupil convergence over the required ROI.
    jobs=[job('Spherical',p,32,d) for p in args.node_pupils for d in (4.8,38.4)]
elif args.suite=='confirm':
    jobs=[job('Spherical',p,32,9.6) for p in (4096,8192)]
    jobs += [job('Spherical',4096,n,d) for n,d in ((32,.6),(64,.3))]
    jobs += [job('Spherical',8192,32,.6)]
    jobs += [job('Spherical',p,32,38.4,pos,x,y,w)
             for pos,x,y,w in (('b0',2.1375,2.1375,.465),('b2',-2.1375,-2.1375,.606))
             for p in (4096,8192)]
elif args.suite=='pixel_patch':
    jobs=[job('Spherical',4096,n,d) for n,d in ((32,.6),(64,.3),(128,.15))]
elif args.suite=='high_reference':
    jobs=[job('Spherical',16384,32,38.4,pos,x,y,w)
          for pos,x,y,w in (('b0',2.1375,2.1375,.465),('b2',-2.1375,-2.1375,.606))]
else:
    jobs=[job('Spherical',p,256,.6,pos,x,y,w) for p in (args.pupil//2,args.pupil)
          for pos in ('a','b0','b1','b2','b3') for x,y in ((0.,0.),(2.1375,2.1375))
          for w in (.465,.535,.606)]
report={'suite':args.suite,'completed':False,'source_model':source,'rows':[],
        'production_modified':False,'intensity_scale':'Strehl-referenced; integral includes delta squared'}
status=OUT/(args.suite+'_status.json')
def save():
    tmp=status.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2),encoding='utf-8');tmp.replace(status)
save()
with connect(cfg.zemax.install_dir) as session:
    session.system.LoadFile(str(ROOT/'runs/div2k_quarter_four_v1/prepared_model.zmx'),False)
    advanced=session.system.SystemData.Advanced
    report['original_huygens_integral_method']=str(advanced.HuygensIntegralMethod)
    for mode,p,n,d,pos,x,y,w in jobs:
        key=f'{mode}_p{p}_i{n}_d{d:g}_{pos}_x{x:g}_y{y:g}_w{w:g}'
        target=OUT/(key+'.npz');meta=OUT/(key+'.json')
        if target.exists() and meta.exists():
            row=json.loads(meta.read_text(encoding='utf-8'))
            if hashlib.sha256(target.read_bytes()).hexdigest()!=row['npz_sha256']:raise RuntimeError('Cached PSF changed')
            report['rows'].append(row);save();continue
        set_lens_shift(session,cfg,*(cfg.dataset.positions_mm[int(pos[1:])] if pos!='a' else (0.,0.)))
        fields=session.system.SystemData.Fields;waves=session.system.SystemData.Wavelengths
        fields.GetField(1).X=x;fields.GetField(1).Y=y;waves.GetWavelength(1).Wavelength=w
        advanced.HuygensIntegralMethod=getattr(session.ZOSAPI.SystemData.HuygensIntegralSettings,mode)
        analysis=session.system.Analyses.New_Analysis(session.ZOSAPI.Analysis.AnalysisIDM.HuygensPsf)
        started=time.monotonic()
        try:
            s=analysis.GetSettings();s.Field.SetFieldNumber(1);s.Wavelength.SetWavelengthNumber(1)
            s.PupilSampleSize=_sample_size(session.ZOSAPI,p);s.ImageSampleSize=_sample_size(session.ZOSAPI,n)
            s.ImageDelta=d;s.Normalize=False;s.UseCentroid=False;s.UsePolarization=False
            s.Type=session.ZOSAPI.Analysis.Settings.HuygensPsfTypes.Linear
            analysis.ApplyAndWaitForCompletion();r=analysis.GetResults()
            if int(r.NumberOfDataGrids)!=1:raise RuntimeError('Missing floating PSF')
            g=r.GetDataGrid(0)
            a=np.fromiter(g.Values,dtype=float,count=int(g.Nx)*int(g.Ny)).reshape(int(g.Ny),int(g.Nx))
            if not np.isfinite(a).all() or a.min()<0 or a.sum()<=0:raise RuntimeError('Invalid PSF')
            dx=float(g.Dx);dy=float(g.Dy)
            if abs(dx-d)>1e-9 or abs(dy-d)>1e-9 or a.shape!=(n,n):raise RuntimeError('Unexpected output grid')
            band=max(1,n//16);outer=1-a[band:-band,band:-band].sum()/a.sum()
            np.savez_compressed(target,psf=a)
            row={'key':key,'method':mode,'pupil':p,'image_n':n,'delta_um':d,'window_um':n*d,'array_axes':'x,y',
                 'position':pos,'object_xy_mm':[x,y],'wavelength_um':w,'sum':float(a.sum()),
                 'integral_relative_um2':float(a.sum()*dx*dy),'peak':float(a.max()),
                 'outer_band_fraction':float(outer),'minx_um':float(g.MinX),'miny_um':float(g.MinY),
                 'elapsed_s':time.monotonic()-started,'npz_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                 'messages':[str(r.GetMessageAt(k)) for k in range(r.NumberOfMessages)]}
            if n<=128:
                import re
                text_path=OUT/(key+'.txt')
                if r.GetTextFile(str(text_path)):
                    text=text_path.read_text(encoding='utf-16' if text_path.read_bytes()[:2] in (b'\xff\xfe',b'\xfe\xff') else 'utf-8-sig',errors='replace')
                    match=re.search(r'Center coordinates\s*:\s*([+\-\d.Ee]+),\s*([+\-\d.Ee]+)',text)
                    if match:row['chief_xy_mm']=[float(match.group(1)),float(match.group(2))]
            meta.write_text(json.dumps(row,indent=2),encoding='utf-8')
            report['rows'].append(row);save();print(json.dumps(row),flush=True)
        finally:analysis.Close()
report['completed']=True;save();print('COMPLETED '+args.suite,flush=True)
