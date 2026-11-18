"""Locked test evaluation; each layout uses its own matching input burst."""
import json
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from super_resolution_imaging.training.dataset import ZemaxBurstDataset
from super_resolution_imaging.training.evaluation import load_model, confidence_interval
from super_resolution_imaging.training.baselines import metrics

root=Path(__file__).resolve().parents[1];torch.set_num_threads(4)
cfg=json.loads((root/'configs/quarter_four_training_frozen.json').read_text())
spec={'quarter_best':('div2k_quarter_four_v1','train_b_quarter_four_frozen_stage2/best.pt','b'),

      'original_best':('div2k_train_v1','train_b_expanded_split_residual_ssim_w020_stage2/best.pt','b'),
      'a_best':('div2k_train_v1','train_a_expanded_split_residual_ssim_w020_stage1/best.pt','a')}
import hashlib
lock=json.loads((root/'runs/quarter_four_prediction_inspection/evaluation_checkpoint.json').read_text())
assert lock['epoch']==34
assert hashlib.sha256(Path(lock['path']).read_bytes()).hexdigest()==lock['sha256']
provenance={}
rows={};ids=None;target_hashes=None
for name,(dataset,checkpoint,group) in spec.items():
    data=ZemaxBurstDataset(root/'runs'/dataset,group,'test',exclude_ids=tuple(cfg['exclude_ids']))
    current=[r['sample_id'] for r in data.records]
    if ids is None:ids=current
    assert ids==current and len(current)==80
    hashes=[r['target_sha256'] for r in data.records]
    if target_hashes is None:target_hashes=hashes
    assert hashes==target_hashes
    provenance[name]={'checkpoint':checkpoint,'sha256':hashlib.sha256((root/'runs'/checkpoint).read_bytes()).hexdigest()}
    model,_=load_model(root/'runs'/checkpoint,group,torch.device('cpu'));values=[]
    with torch.inference_mode():
        for batch in DataLoader(data,batch_size=8):
            scores=metrics(model(batch['input']),batch['target'])
            values.extend(np.array([s.numpy() for s in scores]).T.tolist())
    rows[name]=np.array(values);print(name,rows[name].mean(axis=0).tolist(),flush=True)
report={'split':'test','samples':len(ids),'test_used':True,
        'locked_checkpoint':lock,'checkpoints':provenance,'sample_ids':ids,'metrics':{k:dict(zip(('l1','psnr_db','ssim'),v.mean(axis=0).tolist())) for k,v in rows.items()},'paired':{}}
for reference in ('original_best','a_best'):
    delta=rows['quarter_best'][:,1]-rows[reference][:,1]
    report['paired'][reference]={'mean_psnr_gain':float(delta.mean()),'median':float(np.median(delta)),
          'ci95':confidence_interval(delta),'improved':int((delta>0).sum())}
out=root/'runs/quarter_four_test_evaluation';out.mkdir(exist_ok=True)
with (out/'per_sample.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['sample_id']+[f'{k}_{m}' for k in rows for m in ('l1','psnr_db','ssim')])
    for i,sid in enumerate(ids):w.writerow([sid]+[float(x) for v in rows.values() for x in v[i]])
(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

