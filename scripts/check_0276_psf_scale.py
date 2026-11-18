"""Isolated photometric controls; never modifies the release dataset."""
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
from super_resolution_imaging.config import load_config
from super_resolution_imaging.zos.connection import connect
from super_resolution_imaging.zos.model import set_lens_shift, validate_source
from super_resolution_imaging.zos.image_simulation import run_image_simulation

root = Path(__file__).resolve().parents[1]
out = root / 'runs/0276_03_psf_scale_check'
out.mkdir(exist_ok=True)
cfg = load_config(root / 'configs/zemax_quarter_four_frozen.toml')
validate_source(cfg)
source = np.asarray(Image.open(root/'runs/div2k_quarter_four_v1/samples/0276_03/source.png').convert('L'))
patterns = {'gray064': np.full((768,768),64,np.uint8),
            'gray128': np.full((768,768),128,np.uint8),
            'scene_half': (source.astype(np.uint16)//2).astype(np.uint8)}
# Finite bright patch: effective image response, not a monochromatic optical PSF.
point = np.zeros((768,768),np.uint8)
point[376:392,376:392] = 255
patterns['point_center'] = point
point2 = np.zeros_like(point)
point2[184:200,568:584] = 255
patterns['point_offaxis'] = point2
report = {'completed':False,'dataset_modified':False,'generation_resumed':False,
          'note':'Point patterns measure finite-patch effective responses, not direct optical PSFs.', 'patterns':{}}
def save():
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
with connect(cfg.zemax.install_dir) as session:
    session.system.LoadFile(str(root/'runs/div2k_quarter_four_v1/prepared_model.zmx'),False)
    analysis = session.system.Analyses.New_Analysis(session.ZOSAPI.Analysis.AnalysisIDM.ImageSimulation)
    settings = analysis.GetSettings()
    props = {}
    for name in dir(settings):
        if name.startswith('_'): continue
        try:
            value = getattr(settings,name)
            if not callable(value): props[name] = str(value)
        except Exception as exc: props[name] = str(exc)
    report['default_settings'] = props
    settings.SaveTo(str(out/'default_image_simulation.cfg'))
    analysis.Close()
    save()
    for name, arr in patterns.items():
        path = out/(name+'_input.png')
        Image.fromarray(arr).save(path)
        staged = cfg.zemax.images_dir/('sr_scale_0276_'+name+'.png')
        if staged.exists(): raise FileExistsError(staged)
        shutil.copy2(path,staged)
        rows = []
        try:
            for i, position in enumerate(cfg.dataset.positions_mm):
                target = out/(name+f'_frame_{i:02d}.png')
                if not target.exists():
                    set_lens_shift(session,cfg,*position)
                    run_image_simulation(session,staged.name,target,cfg.simulation)
                with Image.open(target) as im:
                    mode = im.mode
                    data = np.asarray(im.convert('L'),dtype=float)/255
                total = data.sum()
                yy,xx = np.indices(data.shape)
                rows.append({'frame':i,'raw_mode':mode,'mean':float(data.mean()),
                             'interior_mean':float(data[16:-16,16:-16].mean()),
                             'sum':float(total),'max':float(data.max()),
                             'saturated_fraction':float((data==1).mean()),
                             'centroid_x':float((data*xx).sum()/total) if total else None,
                             'centroid_y':float((data*yy).sum()/total) if total else None})
                report['patterns'][name] = rows
                save()
                print(name,i,rows[-1],flush=True)
        finally:
            staged.unlink(missing_ok=True)
report['completed'] = True
save()
print('COMPLETED',flush=True)
