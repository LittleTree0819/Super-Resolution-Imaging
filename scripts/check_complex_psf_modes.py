import json
from pathlib import Path
import numpy as np
from super_resolution_imaging.config import load_config
from super_resolution_imaging.zos.connection import connect
from super_resolution_imaging.zos.rotated_pupil import apply_equivalent_rotation,native_grid

root=Path(__file__).resolve().parents[1];out=root/'runs/psf_low_memory_reference/complex_modes';out.mkdir(parents=True,exist_ok=True)
cfg=load_config(root/'configs/zemax_quarter_four_frozen.toml');rows=[]
with connect(cfg.zemax.install_dir) as session:
    session.system.LoadFile(str(root/'runs/div2k_quarter_four_v1/prepared_model.zmx'),False)
    apply_equivalent_rotation(session,cfg,cfg.dataset.positions_mm[0],[2.1375,2.1375],.465,0.)
    for mode in ('Planar','Spherical'):
        session.system.SystemData.Advanced.HuygensIntegralMethod=getattr(session.ZOSAPI.SystemData.HuygensIntegralSettings,mode)
        arrays={}
        for kind in ('Linear','Real','Imaginary'):
            a,meta=native_grid(session,256,32,38.4,kind,out/f'{mode}_{kind}.txt')
            arrays[kind]=a;rows.append({'mode':mode,'kind':kind,**meta,'min':float(a.min()),'max':float(a.max())})
        estimated=arrays['Real']**2+arrays['Imaginary']**2
        rows.append({'mode':mode,'closure_l1':float(np.abs(estimated-arrays['Linear']).sum()/arrays['Linear'].sum())})
        np.savez_compressed(out/(mode+'.npz'),**arrays)
(out/'report.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(rows,indent=2))
