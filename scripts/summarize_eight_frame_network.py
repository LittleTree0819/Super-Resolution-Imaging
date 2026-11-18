"""Inspect convergence and capacity-controlled outcomes of the completed pilot."""
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]/'runs/eight_frame_quarter_network_pilot'
s=json.loads((root/'summary.json').read_text())
report={k:s[k] for k in ('paired_gains','paired_source_bootstrap','limitations')}
report['curves']={};report['best_means']={}
for seed in (5400,5401,5402):
    for name in ('a','four','new_four','eight','repeat_eight'):
        h=json.loads((root/f'seed{seed}_{name}/history.json').read_text())
        best=max(h,key=lambda r:r['validation']['16']['psnr'])
        report['curves'][f'{seed}_{name}']={
            'epochs':len(h),'train_l1_first':h[0]['train_l1'],'train_l1_last':h[-1]['train_l1'],
            'first_validation_interior_psnr':h[0]['validation']['16']['psnr'],
            'last_validation_interior_psnr':h[-1]['validation']['16']['psnr'],
            'best_epoch':best['epoch'],'best_interior_psnr':best['validation']['16']['psnr'],
            'final_lr':h[-1]['used_learning_rate']}
for name in ('four','new_four','eight','repeat_eight'):
    report['best_means'][name]={metric:float(np.mean([s['results']['best'][name][str(seed)]['metrics']['16'][metric]
        for seed in (5400,5401,5402)])) for metric in ('psnr','ssim')}
(root/'convergence_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
