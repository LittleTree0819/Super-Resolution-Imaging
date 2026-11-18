"""Summarize fusion tradeoffs and GT-assisted brightness diagnostics; no training."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from super_resolution_imaging.training.baselines import shift_and_add, metrics
from super_resolution_imaging.training.evaluation import confidence_interval


def main():
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1]
    out=root/'runs/eight_frame_quarter_analysis'
    rows=list(csv.DictReader((out/'fusion_per_sample.csv').open()))
    summary={'mean_metrics':{},'paired':{},'brightness_oracle_diagnostic':{},'scene_statistics':{}}
    for method in ('four','new_four','eight','repeat_eight'):
        summary['mean_metrics'][method]={k.split(method+'_',1)[1]:float(np.mean([float(r[k]) for r in rows]))
            for k in rows[0] if k.startswith(method+'_')}
    for a,b in [('new_four','four'),('eight','new_four')]:
        for metric in ('psnr_db','ssim'):
            delta=np.array([float(r[f'{a}_interior_{metric}'])-float(r[f'{b}_interior_{metric}']) for r in rows])
            summary['paired'][f'{a}_minus_{b}_{metric}']={'mean':float(delta.mean()),'ci95_mean':confidence_interval(delta),
                'improved':int((delta>0).sum())}
    geometry=json.loads((root/'runs/eight_frame_quarter_geometry/phase_gate.json').read_text())
    shifts=[tuple(geometry['frames'][str(i)]['shift_mean_lr_px']) for i in range(8)]
    per_scene=[]
    def load(p):return torch.from_numpy(np.asarray(Image.open(p),dtype=np.float32)/255.)
    for r in rows:
        sid=r['sample_id'];p=root/'runs/eight_frame_quarter_pilot_32/samples'/sid
        frames=torch.stack([load(p/'b'/f'frame_{i:02d}.png') for i in range(8)])
        target=load(p/'target.png')[None,None]
        pred={'four':shift_and_add(frames[:4][None],tuple(shifts[:4])),
              'new_four':shift_and_add(frames[4:][None],tuple(shifts[4:])),
              'eight':shift_and_add(frames[None],tuple(shifts))}
        row={'sample_id':sid,'target_std':float(target.std()),'target_mean':float(target.mean())}
        for method,values in pred.items():
            a=values[...,16:-16,16:-16];b=target[...,16:-16,16:-16]
            bias=(a-b).mean();corrected=(a-bias).clamp(0,1)
            row[f'{method}_bias']=float(bias)
            row[f'{method}_oracle_psnr_db']=float(metrics(corrected,b)[1][0])
        per_scene.append(row)
    for a,b in [('eight','four'),('new_four','four')]:
        delta=np.array([r[f'{a}_oracle_psnr_db']-r[f'{b}_oracle_psnr_db'] for r in per_scene])
        summary['brightness_oracle_diagnostic'][f'{a}_minus_{b}']={'mean_psnr_gain_db':float(delta.mean()),
            'ci95_mean':confidence_interval(delta),'improved':int((delta>0).sum()),
            'warning':'Uses GT interior mean; diagnostic only, not usable at inference or as official metric'}
    worst=min(rows,key=lambda r:float(r['eight_interior_psnr_db'])-float(r['four_interior_psnr_db']))
    summary['worst_scene']={**worst,**next(r for r in per_scene if r['sample_id']==worst['sample_id'])}
    summary['scene_statistics']['target_std_min']=min(r['target_std'] for r in per_scene)
    delta=np.array([float(r['eight_interior_psnr_db'])-float(r['four_interior_psnr_db']) for r in rows])
    summary['interior_delta_median_db']=float(np.median(delta))
    summary['interior_delta_range_db']=[float(delta.min()),float(delta.max())]
    (out/'supplemental_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (out/'brightness_diagnostic.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(per_scene[0]));w.writeheader();w.writerows(per_scene)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
