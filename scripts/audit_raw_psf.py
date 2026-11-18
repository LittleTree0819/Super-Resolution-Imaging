"""Direct PSF convergence with a fixed physical image grid and chief-ray positions."""
import json
from pathlib import Path
import numpy as np
from super_resolution_imaging.config import load_config
from super_resolution_imaging.zos.connection import connect
from super_resolution_imaging.zos.model import set_lens_shift, validate_source
from super_resolution_imaging.zos.image_simulation import _sample_size

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/photometric_forward_audit_v1/raw_psf'
OUT.mkdir(parents=True,exist_ok=True)
cfg=load_config(ROOT/'configs/zemax_quarter_four_frozen.toml');validate_source(cfg)
positions={'a':(0.,0.),'b0':cfg.dataset.positions_mm[0],'b1':cfg.dataset.positions_mm[1]}
field_points={'center':(0.,0.),'edge':(2.1375,2.1375),'opposite':(-2.1375,-2.1375)}
report={'completed':False,'normalize':False,'image_delta_um':1.2,
        'scale_note':'Normalize=False is Strehl-referenced, NOT absolute radiometry; separately unit-sum kernels are shape-only controls.',
        'rows':[]}
def save():
    path=OUT/'report.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2),encoding='utf-8');tmp.replace(path)
with connect(cfg.zemax.install_dir) as session:
    session.system.LoadFile(str(ROOT/'runs/div2k_quarter_four_v1/prepared_model.zmx'),False)
    fields=session.system.SystemData.Fields;waves=session.system.SystemData.Wavelengths
    report['original_fields']=[(float(fields.GetField(i).X),float(fields.GetField(i).Y)) for i in range(1,int(fields.NumberOfFields)+1)]
    report['original_wavelength_um']=float(waves.GetWavelength(1).Wavelength)
    jobs=[(n,128,name,field_name,.535) for n in (128,256,512) for name in positions for field_name in field_points]
    jobs += [(512,256,'a','center',.535)]
    # RGB weights match the grayscale conversion used by the original PNG pipeline.
    jobs += [(512,128,name,field_name,w) for w in (.606,.465) for name in positions for field_name in field_points]
    for n,image_n,name,field_name,wave in jobs:
        key=f'{name}_{field_name}_p{n}_i{image_n}_w{wave:.3f}'
        target=OUT/(key+'.npz');meta=OUT/(key+'.json')
        if target.exists() and meta.exists():
            row=json.loads(meta.read_text(encoding='utf-8'));report['rows'].append(row);save();continue
        set_lens_shift(session,cfg,*positions[name]);x,y=field_points[field_name]
        fields.GetField(1).X=x;fields.GetField(1).Y=y;waves.GetWavelength(1).Wavelength=wave
        analysis=session.system.Analyses.New_Analysis(session.ZOSAPI.Analysis.AnalysisIDM.HuygensPsf)
        try:
            s=analysis.GetSettings();s.Field.SetFieldNumber(1);s.Wavelength.SetWavelengthNumber(1)
            s.PupilSampleSize=_sample_size(session.ZOSAPI,n);s.ImageSampleSize=_sample_size(session.ZOSAPI,image_n)
            s.ImageDelta=1.2;s.Normalize=False;s.UseCentroid=False;s.UsePolarization=False
            s.Type=session.ZOSAPI.Analysis.Settings.HuygensPsfTypes.Linear
            analysis.ApplyAndWaitForCompletion();r=analysis.GetResults()
            if r.NumberOfDataGrids!=1:raise RuntimeError('No direct floating PSF grid')
            g=r.GetDataGrid(0);a=np.fromiter(g.Values,dtype=float,count=int(g.Nx)*int(g.Ny)).reshape(int(g.Ny),int(g.Nx))
            if not np.isfinite(a).all() or a.min()<0 or a.sum()<=0:raise RuntimeError('Invalid PSF')
            kernel=a/a.sum();band=max(1,image_n//16);inside=kernel[band:-band,band:-band].sum()
            row={'key':key,'position':name,'field':field_name,'object_xy_mm':[x,y],
                 'pupil_sampling':n,'image_sampling':image_n,'wavelength_um':wave,
                 'dx_um':float(g.Dx),'dy_um':float(g.Dy),'minx_um':float(g.MinX),'miny_um':float(g.MinY),
                 'sum_strehl_referenced':float(a.sum()),'peak_strehl_referenced':float(a.max()),
                 'outer_band_energy_fraction':float(1-inside),
                 'messages':[str(r.GetMessageAt(i)) for i in range(r.NumberOfMessages)]}
            np.savez_compressed(target,psf=a,kernel=kernel)
            meta.write_text(json.dumps(row,indent=2),encoding='utf-8')
            report['rows'].append(row);save();print(json.dumps(row),flush=True)
        finally:analysis.Close()
report['completed']=True;save();print('COMPLETED',flush=True)
