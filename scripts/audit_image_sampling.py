"""Isolated/resumable Image Simulation sampling audit; never writes a dataset."""
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
import numpy as np
from PIL import Image
from super_resolution_imaging.config import load_config
from super_resolution_imaging.zos.connection import connect
from super_resolution_imaging.zos.model import set_lens_shift, validate_source
from super_resolution_imaging.zos.image_simulation import run_image_simulation

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/photometric_forward_audit_v1/image_sampling'
OUT.mkdir(parents=True,exist_ok=True)
cfg=load_config(ROOT/'configs/zemax_quarter_four_frozen.toml')
validate_source(cfg)
sample=ROOT/'runs/div2k_quarter_four_v1/samples/0276_03'
report={'completed':False,'dataset_modified':False,'training_started':False,'rows':[]}
def save():
    p=OUT/'report.json';t=p.with_suffix('.tmp')
    t.write_text(json.dumps(report,indent=2),encoding='utf-8');t.replace(p)
def image(path):return np.asarray(Image.open(path).convert('L'),dtype=float)/255
jobs=[]
for n in (128,256,512):
    for frame in (0,1):
        cached=sample/'b'/f'frame_{frame:02d}.png' if n==128 else ROOT/'runs/0276_03_psf_scale_check/followup'/f'scene_sampling256_{frame:02d}.png' if n==256 else None
        jobs.append((f'b{frame}',frame,n,False,cached))
for n in (128,256,512):
    jobs.append(('a',None,n,False,sample/'a/frame_00.png' if n==128 else None))
for n in (128,256,512):
    jobs.append(('gt',None,n,True,sample/'gt.png' if n==128 and (sample/'gt.png').exists() else None))
with connect(cfg.zemax.install_dir) as session:
    session.system.LoadFile(str(ROOT/'runs/div2k_quarter_four_v1/prepared_model.zmx'),False)
    staged=cfg.zemax.images_dir/'sr_forward_sampling_0276.png'
    if staged.exists():raise FileExistsError(staged)
    shutil.copy2(sample/'source.png',staged)
    try:
        for name,frame,n,ideal,cached in jobs:
            target=OUT/f'{name}_{n}.png';started=time.monotonic()
            if not target.exists():
                if cached is not None and cached.exists():shutil.copy2(cached,target)
                else:
                    set_lens_shift(session,cfg,*(cfg.dataset.positions_mm[frame] if frame is not None else (0.,0.)))
                    sim=replace(cfg.simulation,pupil_sampling=n,image_sampling=n)
                    run_image_simulation(session,staged.name,target,sim,ideal=ideal)
            a=image(target)
            row={'group':name,'sampling':n,'ideal':ideal,'shape':list(a.shape),'mean':float(a.mean()),
                 'max':float(a.max()),'elapsed_this_run_s':time.monotonic()-started,'cached':cached is not None}
            previous=OUT/f'{name}_{n//2}.png'
            if previous.exists():
                b=image(previous);row.update(mae_vs_previous=float(np.abs(a-b).mean()),rmse_vs_previous=float(np.sqrt(np.mean((a-b)**2))))
            report['rows'].append(row);save();print(json.dumps(row),flush=True)
    finally:staged.unlink(missing_ok=True)
report['completed']=True;save()
print('COMPLETED',flush=True)
