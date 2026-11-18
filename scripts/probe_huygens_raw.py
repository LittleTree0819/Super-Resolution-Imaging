"""Read a direct floating-point Huygens PSF; no PNG normalization."""
import json
from pathlib import Path
import numpy as np
from super_resolution_imaging.config import load_config
from super_resolution_imaging.zos.connection import connect
from super_resolution_imaging.zos.model import set_lens_shift
from super_resolution_imaging.zos.image_simulation import _sample_size

root=Path(__file__).resolve().parents[1];out=root/'runs/photometric_forward_audit_v1/raw_probe'
out.mkdir(parents=True,exist_ok=True)
cfg=load_config(root/'configs/zemax_quarter_four_frozen.toml')
with connect(cfg.zemax.install_dir) as session:
    session.system.LoadFile(str(root/'runs/div2k_quarter_four_v1/prepared_model.zmx'),False)
    set_lens_shift(session,cfg,0.,0.)
    fields=session.system.SystemData.Fields;waves=session.system.SystemData.Wavelengths
    report={'original_field_type':str(fields.GetFieldType()),
            'original_fields':[{'x':float(fields.GetField(i).X),'y':float(fields.GetField(i).Y)} for i in range(1,int(fields.NumberOfFields)+1)],
            'original_wavelengths':[float(waves.GetWavelength(i).Wavelength) for i in range(1,int(waves.NumberOfWavelengths)+1)]}
    fields.GetField(1).X=0.;fields.GetField(1).Y=0.;waves.GetWavelength(1).Wavelength=.535
    analysis=session.system.Analyses.New_Analysis(session.ZOSAPI.Analysis.AnalysisIDM.HuygensPsf)
    try:
        s=analysis.GetSettings();s.Field.SetFieldNumber(1);s.Wavelength.SetWavelengthNumber(1)
        s.PupilSampleSize=_sample_size(session.ZOSAPI,128);s.ImageSampleSize=_sample_size(session.ZOSAPI,128)
        s.ImageDelta=1.2;s.Normalize=False;s.UseCentroid=False;s.UsePolarization=False
        s.Type=session.ZOSAPI.Analysis.Settings.HuygensPsfTypes.Linear
        analysis.ApplyAndWaitForCompletion();r=analysis.GetResults()
        report['header']=[str(x) for x in r.HeaderData.Lines]
        report['grid_count']=int(r.NumberOfDataGrids)
        report['messages']=[str(r.GetMessageAt(i)) for i in range(r.NumberOfMessages)]
        if r.NumberOfDataGrids!=1:raise RuntimeError('Expected a single floating PSF grid')
        g=r.GetDataGrid(0);nx=int(g.Nx);ny=int(g.Ny)
        a=np.fromiter(g.Values,dtype=float,count=nx*ny).reshape(ny,nx)
        np.save(out/'psf.npy',a)
        report['grid']={'nx':nx,'ny':ny,'dx':float(g.Dx),'dy':float(g.Dy),'minx':float(g.MinX),'miny':float(g.MinY),
                        'description':str(g.Description),'xlabel':str(g.XLabel),'ylabel':str(g.YLabel),'value_label':str(g.ValueLabel),
                        'sum':float(a.sum()),'max':float(a.max()),'min':float(a.min()),'finite':bool(np.isfinite(a).all())}
        r.GetTextFile(str(out/'psf.txt'))
    finally:analysis.Close()
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
