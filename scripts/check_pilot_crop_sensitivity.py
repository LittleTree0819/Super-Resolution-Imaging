"""Check whether apparent nine-frame gain persists away from image boundaries."""
import json
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from super_resolution_imaging.training.baselines import shift_and_add

root=Path('runs/nine_frame_pilot_32')
out=Path('runs/nine_frame_pilot_diagnostics')
p=argparse.ArgumentParser()
p.add_argument('--calibration',type=Path,default=Path('runs/nine_frame_calibration_refined/shift_report.json'))
p.add_argument('--output',type=Path,default=out/'crop_sensitivity.json')
args=p.parse_args()
cal=json.loads(args.calibration.read_text())
shifts=tuple(tuple(cal['frames'][str(i)]['median_measured_shift_xy']) for i in range(9))
records=[json.loads(line) for line in (root/'manifest.jsonl').read_text().splitlines()]
four=(0,1,7,8)
rows=[]
torch.set_num_threads(4)
with torch.inference_mode():
    for rec in records:
        folder=root/'samples'/rec['sample_id']
        frames=torch.stack([torch.from_numpy(np.asarray(Image.open(folder/'b'/f'frame_{i:02d}.png'),dtype=np.float32)/255) for i in range(9)])[None]
        target=torch.from_numpy(np.asarray(Image.open(folder/'target.png'),dtype=np.float32)/255)[None,None]
        a=shift_and_add(frames[:,four],tuple(shifts[i] for i in four))
        b=shift_and_add(frames,shifts)
        for margin in (0,2,4,8,16,32):
            sl=slice(margin,-margin) if margin else slice(None)
            mse4=(a[:,:,sl,sl]-target[:,:,sl,sl]).square().mean()
            mse9=(b[:,:,sl,sl]-target[:,:,sl,sl]).square().mean()
            rows.append({'sample_id':rec['sample_id'],'hr_margin':margin,'nine_minus_four_psnr':float(10*torch.log10(mse4/mse9))})
rng=np.random.default_rng(5400)
summary={}
for margin in (0,2,4,8,16,32):
    values=np.array([r['nine_minus_four_psnr'] for r in rows if r['hr_margin']==margin])
    boots=rng.choice(values,(10000,len(values)),replace=True).mean(1)
    summary[str(margin)]={'mean_gain_db':float(values.mean()),'median_gain_db':float(np.median(values)),'positive_samples':int((values>0).sum()),'bootstrap_mean_95_interval_db':np.percentile(boots,[2.5,97.5]).tolist()}
args.output.write_text(json.dumps({'summary':summary,'rows':rows},indent=2)+'\n')
print(json.dumps(summary,indent=2))
