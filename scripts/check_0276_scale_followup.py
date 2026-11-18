"""Two extreme positions: additivity and PSF sampling sensitivity controls."""
import json
import shutil
from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image
from super_resolution_imaging.config import load_config
from super_resolution_imaging.zos.connection import connect
from super_resolution_imaging.zos.model import set_lens_shift
from super_resolution_imaging.zos.image_simulation import run_image_simulation

root=Path(__file__).resolve().parents[1]
out=root/'runs/0276_03_psf_scale_check/followup';out.mkdir(parents=True,exist_ok=True)
cfg=load_config(root/'configs/zemax_quarter_four_frozen.toml')
sample=root/'runs/div2k_quarter_four_v1/samples/0276_03'
source=np.asarray(Image.open(sample/'source.png').convert('L'))
report={'completed':False,'cases':{}}
with connect(cfg.zemax.install_dir) as session:
    session.system.LoadFile(str(root/'runs/div2k_quarter_four_v1/prepared_model.zmx'),False)
    for name,arr,sim in [('scene_offset',(source.astype(np.uint16)//2+64).astype(np.uint8),cfg.simulation),
                         ('scene_sampling256',source,replace(cfg.simulation,pupil_sampling=256,image_sampling=256))]:
        staged=cfg.zemax.images_dir/('sr_0276_'+name+'.png')
        if staged.exists():raise FileExistsError(staged)
        Image.fromarray(arr).save(staged)
        rows=[]
        try:
            for i in (0,1):
                target=out/(name+f'_{i:02d}.png')
                if not target.exists():
                    set_lens_shift(session,cfg,*cfg.dataset.positions_mm[i])
                    run_image_simulation(session,staged.name,target,sim)
                a=np.asarray(Image.open(target).convert('L'),dtype=float)/255
                original=np.asarray(Image.open(sample/'b'/f'frame_{i:02d}.png').convert('L'),dtype=float)/255
                row={'frame':i,'mean':float(a.mean()),'max':float(a.max()),'mae_vs_original':float(np.abs(a-original).mean())}
                if name=='scene_offset':
                    half=np.asarray(Image.open(out.parent/f'scene_half_frame_{i:02d}.png').convert('L'),dtype=float)/255
                    gray=np.asarray(Image.open(out.parent/f'gray064_frame_{i:02d}.png').convert('L'),dtype=float)/255
                    row['additivity_mae']=float(np.abs(a-half-gray).mean())
                    row['additivity_bias']=float((a-half-gray).mean())
                rows.append(row);report['cases'][name]=rows
                (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                print(name,row,flush=True)
        finally:staged.unlink(missing_ok=True)
report['completed']=True
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
