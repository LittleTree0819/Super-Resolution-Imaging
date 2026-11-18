"""Isolated forward/reverse-order rerenders; original dataset remains unchanged."""
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from super_resolution_imaging.config import load_config
from super_resolution_imaging.zos.connection import connect
from super_resolution_imaging.zos.model import set_lens_shift,validate_source
from super_resolution_imaging.zos.image_simulation import run_image_simulation
from super_resolution_imaging.pipeline import _normalize_grayscale
from super_resolution_imaging.data.writer import sha256_file

root=Path(__file__).resolve().parents[1]
cfg=load_config(root/'configs/zemax_quarter_four_frozen.toml');validate_source(cfg)
sample=root/'runs/div2k_quarter_four_v1/samples/0276_03'
out=root/'runs/0276_03_photometry_recheck';out.mkdir(exist_ok=True)
def values(path):return np.asarray(Image.open(path).convert('L'),dtype=np.float64)/255.
original=[values(sample/'b'/f'frame_{i:02d}.png') for i in range(4)]
report={'original':[{'frame':i,'mean':float(v.mean()),'min':float(v.min()),'max':float(v.max()),
    'interior_mean':float(v[16:-16,16:-16].mean())} for i,v in enumerate(original)],'passes':{},
    'dataset_modified':False,'generation_resumed':False}
for label,order in [('forward',(0,1,2,3)),('reverse',(3,2,1,0))]:
    folder=out/label;folder.mkdir(exist_ok=True)
    with connect(cfg.zemax.install_dir) as session:
        session.system.LoadFile(str(root/'runs/div2k_quarter_four_v1/prepared_model.zmx'),False)
        staged=cfg.zemax.images_dir/('sr_0276_recheck_'+label+'.png')
        if staged.exists():raise FileExistsError(staged)
        shutil.copy2(sample/'source.png',staged)
        try:
            for i in order:
                path=folder/f'frame_{i:02d}.png'
                if path.exists():continue
                set_lens_shift(session,cfg,*cfg.dataset.positions_mm[i])
                run_image_simulation(session,staged.name,path,cfg.simulation)
                _normalize_grayscale(path)
                print(f'{label} frame {i} completed',flush=True)
        finally:staged.unlink(missing_ok=True)
    rows=[]
    for i in range(4):
        path=folder/f'frame_{i:02d}.png';v=values(path)
        rows.append({'frame':i,'mean':float(v.mean()),'same_file_hash':sha256_file(path)==sha256_file(sample/'b'/f'frame_{i:02d}.png'),
            'mean_absolute_difference':float(np.abs(v-original[i]).mean()),'max_absolute_difference':float(np.abs(v-original[i]).max())})
    report['passes'][label]=rows
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
report['completed']=True
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2),flush=True)
