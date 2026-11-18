"""Summarize completed overfit and matched-update validation diagnostics."""
import json
from pathlib import Path
import numpy as np

p=Path(__file__).resolve().parents[1]/'runs/eight_frame_update_checks'
s=json.loads((p/'summary.json').read_text())
report={'mean_metrics':{},'paired_per_source':[],'best_secondary':{},'curves':{}}
for name,values in s['results'].items():
    report['mean_metrics'][name]={margin:{k:float(np.mean([r['metrics'][margin][k] for r in values.values()]))
        for k in ('l1','psnr_db','ssim','hf_rmse')} for margin in ('0','16')}
    best=[json.loads((p/f'seed{seed}_{name}/best_evaluation.json').read_text()) for seed in values]
    report['best_secondary'][name]={k:float(np.mean([r['metrics']['16'][k] for r in best])) for k in ('l1','psnr_db','ssim','hf_rmse')}
ids=[r['sample_id'] for r in s['results']['eight']['5400']['per_sample'] if r['margin']==16]
for sid in ids:
    row={'sample_id':sid}
    for key in ('psnr_db','ssim','l1','hf_rmse'):
        delta=[]
        for seed in ('5400','5401','5402'):
            def get(name):return next(r[key] for r in s['results'][name][seed]['per_sample'] if r['margin']==16 and r['sample_id']==sid)
            delta.append(get('eight')-get('repeat_eight'))
        row[f'delta_{key}']=float(np.mean(delta))
    report['paired_per_source'].append(row)
for folder in p.iterdir():
    if folder.is_dir() and (folder/'history.json').exists():
        h=json.loads((folder/'history.json').read_text())
        best=max(h,key=lambda r:r['metrics']['16']['psnr_db'])
        report['curves'][folder.name]={'last_update':h[-1]['update'],
            'last_train_batch_l1':h[-1]['last_batch_train_l1'],
            'last_interior_psnr_db':h[-1]['metrics']['16']['psnr_db'],
            'last_lr':h[-1]['used_learning_rate'],'best_checked_update':best['update'],
            'best_checked_interior_psnr_db':best['metrics']['16']['psnr_db']}
(p/'result_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
