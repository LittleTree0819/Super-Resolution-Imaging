"""Supplement the completed experiment with validation-only paired diagnostics."""
import csv
import json
from pathlib import Path
import numpy as np
from super_resolution_imaging.training.evaluation import confidence_interval


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'runs/brightness_comparison_20261011'
    def read(name):
        rows=list(csv.DictReader((out/f'validation_{name}.csv').open(encoding='utf-8')))
        return {r['sample_id']:{k:float(v) for k,v in r.items() if k!='sample_id'} for r in rows}
    initial=read('initial'); best=read('C2_best')
    assert set(initial)==set(best)
    result={'C2_best_minus_initial':{},'diagnostic_previous_degraded_subset':{}}
    for metric in next(iter(initial.values())):
        delta=np.array([best[s][metric]-initial[s][metric] for s in sorted(initial)])
        result['C2_best_minus_initial'][metric]={'mean':float(delta.mean()),
            'ci95_mean':confidence_interval(delta),'better':int(((delta>0) if metric in ('psnr_db','ssim') else (delta<0)).sum())}
    previous=list(csv.DictReader((root/'runs/validation_brightness_analysis_20261011/per_sample.csv').open()))
    subset=[r['sample_id'] for r in previous if float(r['delta_psnr_db'])<0]
    for name in ('C0_last','C1_last','C2_last','C3_last','C2_best'):
        rows=read(name)
        result['diagnostic_previous_degraded_subset'][name]={
            'samples':len(subset),
            'mean_psnr_change_vs_initial':float(np.mean([rows[s]['psnr_db']-initial[s]['psnr_db'] for s in subset])),
            'psnr_better':sum(rows[s]['psnr_db']>initial[s]['psnr_db'] for s in subset)}
    (out/'supplemental_analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
