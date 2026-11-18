import json
from pathlib import Path
import numpy as np
from super_resolution_imaging.config import load_config
from super_resolution_imaging.zos.connection import connect
from super_resolution_imaging.zos.rotated_pupil import apply_equivalent_rotation,rotation,trace_probe,native_complex_roi

root=Path(__file__).resolve().parents[1];out=root/'runs/psf_low_memory_reference';out.mkdir(exist_ok=True)
cfg=load_config(root/'configs/zemax_quarter_four_frozen.toml');rows=[]
with connect(cfg.zemax.install_dir) as session:
    session.system.LoadFile(str(root/'runs/div2k_quarter_four_v1/prepared_model.zmx'),False)
    for angle in (0.,45.):
        field=np.array([2.1375,2.1375])
        apply_equivalent_rotation(session,cfg,cfg.dataset.positions_mm[0],field,.465,angle)
        chief=trace_probe(session,rotation(angle)@field,np.array([[0.,0.]]))[0,2:4]
        e,lin,meta=native_complex_roi(session,1024,32,38.4,chief)
        rows.append({'angle':angle,**meta});np.savez_compressed(out/f'reuse_probe_{angle:g}.npz',field=e,linear=lin)
(out/'reuse_probe.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(rows,indent=2))
