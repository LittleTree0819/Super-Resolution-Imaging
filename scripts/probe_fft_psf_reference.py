"""Independent FFT convergence reference; not approved as a production PSF."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from super_resolution_imaging.config import load_config
from super_resolution_imaging.zos.connection import connect
from super_resolution_imaging.zos.model import set_lens_shift,validate_source

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/psf_reliability_v2/fft_reference'
OUT.mkdir(parents=True,exist_ok=True)
parser=argparse.ArgumentParser();parser.add_argument('--pupils',type=int,nargs='+',default=[512,1024,2048]);parser.add_argument('--output',type=int);args=parser.parse_args()
if args.output:
    OUT=OUT/('output_'+str(args.output));OUT.mkdir(parents=True,exist_ok=True)
cfg=load_config(ROOT/'configs/zemax_quarter_four_frozen.toml');validate_source(cfg)
report={'completed':False,'reference_only':True,'rows':[],'failures':[],'note':'FFT assumptions require validation against Huygens; no algorithm substitution.'}
def save():
    path=OUT/'report.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2),encoding='utf-8');tmp.replace(path)
with connect(cfg.zemax.install_dir) as session:
    session.system.LoadFile(str(ROOT/'runs/div2k_quarter_four_v1/prepared_model.zmx'),False)
    set_lens_shift(session,cfg,0.,0.)
    fields=session.system.SystemData.Fields;fields.GetField(1).X=0.;fields.GetField(1).Y=0.
    session.system.SystemData.Wavelengths.GetWavelength(1).Wavelength=.535
    for n in args.pupils:
        target=OUT/f'a_center_p{n}.npz';meta=target.with_suffix('.json')
        if target.exists() and meta.exists():
            row=json.loads(meta.read_text(encoding='utf-8'));report['rows'].append(row);save();continue
        analysis=session.system.Analyses.New_Analysis(session.ZOSAPI.Analysis.AnalysisIDM.FftPsf)
        started=time.monotonic()
        try:
            s=analysis.GetSettings();s.Field.SetFieldNumber(1);s.Wavelength.SetWavelengthNumber(1)
            sampling=session.ZOSAPI.Analysis.Settings.Psf.PsfSampling
            out_n=args.output or 2*n
            s.SampleSize=getattr(sampling,f'PsfS_{n}x{n}');s.OutputSize=getattr(sampling,f'PsfS_{out_n}x{out_n}')
            s.ImageDelta=-1.;s.Normalize=False;s.UsePolarization=False
            s.Type=session.ZOSAPI.Analysis.Settings.Psf.FftPsfType.Linear
            analysis.ApplyAndWaitForCompletion();r=analysis.GetResults()
            if int(r.NumberOfDataGrids)!=1:raise RuntimeError('Missing FFT numeric grid')
            g=r.GetDataGrid(0);ny=int(g.Ny);nx=int(g.Nx)
            if nx==0 or ny==0:
                errors=[]
                for k in range(r.NumberOfMessages):
                    msg=r.GetMessageAt(k);record={}
                    for name in dir(msg):
                        if name.startswith('_'):continue
                        try:
                            value=getattr(msg,name)
                            if not callable(value):record[name]=str(value)
                        except Exception:pass
                    errors.append(record)
                failure={'pupil':n,'empty_grid':[ny,nx],'messages':errors,'header':[str(x) for x in r.HeaderData.Lines]}
                (OUT/'failure.json').write_text(json.dumps(failure,indent=2),encoding='utf-8')
                r.GetTextFile(str(OUT/'failure.txt'))
                print(json.dumps(failure),flush=True)
                report['failures'].append(failure);save()
                continue
            a=np.fromiter(g.Values,dtype=float,count=nx*ny).reshape(ny,nx)
            if not np.isfinite(a).all() or a.min()<0 or a.sum()<=0:raise RuntimeError('Invalid FFT PSF')
            total=a.sum();band=max(1,nx//16)
            np.savez_compressed(target,psf=a)
            row={'pupil':n,'image_n':nx,'array_axes':'x,y','dx_um':float(g.Dx),'dy_um':float(g.Dy),
                 'minx_um':float(g.MinX),'miny_um':float(g.MinY),'sum':float(total),
                 'integral_relative_um2':float(total*g.Dx*g.Dy),'peak':float(a.max()),
                 'outer_band_fraction':float(1-a[band:-band,band:-band].sum()/total),
                 'elapsed_s':time.monotonic()-started,'npz_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                 'messages':[str(r.GetMessageAt(k)) for k in range(r.NumberOfMessages)]}
            meta.write_text(json.dumps(row,indent=2),encoding='utf-8');report['rows'].append(row);save();print(json.dumps(row),flush=True)
        finally:analysis.Close()
report['completed']=True;save();print('COMPLETED FFT reference',flush=True)
