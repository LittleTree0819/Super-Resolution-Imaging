"""Fit input-only brightness references on training sources; assess on validation."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from super_resolution_imaging.training.dataset import ZemaxBurstDataset, DEFAULT_EXCLUDED_IDS


def main():
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1]
    out=root/'runs/brightness_reference_audit_20261011'
    out.mkdir(exist_ok=False)
    rows=[]
    for split in ('train','validation'):
        ds=ZemaxBurstDataset(root/'runs/div2k_train_v1','b',split,exclude_ids=DEFAULT_EXCLUDED_IDS)
        for index, record in enumerate(ds.records):
            sample=ds[index]
            means=sample['input'].mean(dim=(1,2)).numpy()
            rows.append({'sample_id':record['sample_id'],'split':split,'source_id':record['sample_id'].split('_')[0],
                         'b0_mean':float(means[0]), 'burst_mean':float(means.mean()),
                         'frame_mean_range':float(np.ptp(means)), 'target_mean':float(sample['target'].mean())})
        print(split,len(ds),flush=True)
    train=[r for r in rows if r['split']=='train']; val=[r for r in rows if r['split']=='validation']
    sources=sorted({r['source_id'] for r in train})
    rng=np.random.default_rng(5400); rng.shuffle(sources)
    folds={sid:i%5 for i,sid in enumerate(sources)}
    def fit(records,key):
        x=np.array([r[key] for r in records]); y=np.array([r['target_mean'] for r in records])
        return np.linalg.lstsq(np.column_stack((x,np.ones(len(x)))),y,rcond=None)[0]
    def predict(records,key,coeff): return np.clip(coeff[0]*np.array([r[key] for r in records])+coeff[1],0,1)
    def score(pred,records):
        error=pred-np.array([r['target_mean'] for r in records])
        return {'mae':float(np.abs(error).mean()), 'rmse':float(np.sqrt(np.mean(error**2))),
                'bias':float(error.mean()), 'p95_abs_error':float(np.percentile(np.abs(error),95))}
    report={'train_samples':len(train),'validation_samples':len(val),'source_grouped_cv_folds':5,
            'references':{}}
    for key in ('b0_mean','burst_mean'):
        cv=np.zeros(len(train))
        for fold in range(5):
            fitting=[r for r in train if folds[r['source_id']]!=fold]
            indices=[i for i,r in enumerate(train) if folds[r['source_id']]==fold]
            cv[indices]=predict([train[i] for i in indices],key,fit(fitting,key))
        coeff=fit(train,key)
        pred=predict(val,key,coeff)
        report['references'][key]={'raw_train':score(np.array([r[key] for r in train]),train),
                                   'raw_validation':score(np.array([r[key] for r in val]),val),
                                   'affine_coefficients':coeff.tolist(), 'grouped_cv':score(cv,train),
                                   'affine_validation':score(pred,val)}
        for i,r in enumerate(val): r[f'{key}_affine_reference']=float(pred[i])
    chosen=min(report['references'],key=lambda k:report['references'][k]['grouped_cv']['mae'])
    report['chosen_by_training_cv']=chosen
    diagnostics={r['sample_id']:r for r in csv.DictReader((root/'runs/validation_brightness_analysis_20261011/per_sample.csv').open())}
    for name in ('old_b','new_b'):
        mean_prediction=np.array([float(diagnostics[r['sample_id']]['target_mean'])+float(diagnostics[r['sample_id']][f'{name}_bias']) for r in val])
        report[f'{name}_learned_output_mean']=score(mean_prediction,val)
    report['frame_mean_range']={s:float(np.mean([r['frame_mean_range'] for r in rows if r['split']==s])) for s in ('train','validation')}
    with (out/'validation_references.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(val[0]));w.writeheader();w.writerows(val)
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
