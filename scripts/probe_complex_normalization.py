import json
from pathlib import Path
import numpy as np
from super_resolution_imaging.config import load_config
from super_resolution_imaging.zos.connection import connect
from super_resolution_imaging.zos.rotated_pupil import apply_equivalent_rotation
from super_resolution_imaging.zos.image_simulation import _sample_size

root=Path(__file__).resolve().parents[1];out=root/'runs/psf_low_memory_reference/normalization';out.mkdir(parents=True,exist_ok=True)
cfg=load_config(root/'configs/zemax_quarter_four_frozen.toml');report=[]
with connect(cfg.zemax.install_dir) as session:
    session.system.LoadFile(str(root/'runs/div2k_quarter_four_v1/prepared_model.zmx'),False)
    apply_equivalent_rotation(session,cfg,cfg.dataset.positions_mm[0],[2.1375,2.1375],.465,0.)
    for mode in ('Planar','Spherical'):
        session.system.SystemData.Advanced.HuygensIntegralMethod=getattr(session.ZOSAPI.SystemData.HuygensIntegralSettings,mode)
        for normalized in (False,True):
            arrays={}
            for kind in ('Linear','Real','Imaginary'):
                analysis=session.system.Analyses.New_Analysis_SettingsFirst(session.ZOSAPI.Analysis.AnalysisIDM.HuygensPsf)
                try:
                    s=analysis.GetSettings();s.Field.SetFieldNumber(1);s.Wavelength.SetWavelengthNumber(1)
                    s.PupilSampleSize=_sample_size(session.ZOSAPI,256);s.ImageSampleSize=_sample_size(session.ZOSAPI,32)
                    s.ImageDelta=38.4;s.Type=getattr(session.ZOSAPI.Analysis.Settings.HuygensPsfTypes,kind)
                    s.Normalize=normalized;s.UseCentroid=False;s.UsePolarization=False
                    analysis.ApplyAndWaitForCompletion();analysis.WaitForCompletion();r=analysis.GetResults();g=r.GetDataGrid(0)
                    a=np.fromiter(g.Values,dtype=float,count=int(g.Nx)*int(g.Ny)).reshape(int(g.Nx),int(g.Ny));arrays[kind]=a
                    row={'mode':mode,'normalize':normalized,'kind':kind,'actual_type':str(s.Type),'actual_normalize':bool(s.Normalize),
                         'show_as':str(s.ShowAsType),'finite':bool(np.isfinite(a).all()),
                         'min':float(np.nanmin(a)),'max':float(np.nanmax(a)),
                         'messages':[str(r.GetMessageAt(k).Text) for k in range(r.NumberOfMessages)]}
                    report.append(row);print(row,flush=True)
                finally:analysis.Close()
            intensity=arrays['Real']**2+arrays['Imaginary']**2
            error=float(np.abs(intensity-arrays['Linear']).sum()/arrays['Linear'].sum())
            report.append({'mode':mode,'normalize':normalized,'closure':error})
            np.savez_compressed(out/f'{mode}_{normalized}.npz',**arrays)
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
