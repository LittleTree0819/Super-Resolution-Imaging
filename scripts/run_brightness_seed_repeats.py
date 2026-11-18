"""Repeat C0/C2 and gate a locked, one-time test evaluation on validation stability."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from run_brightness_comparison import evaluate, averages, write_json
from super_resolution_imaging.training.evaluation import load_model, confidence_interval
from super_resolution_imaging.training.cli import sha256_file
from super_resolution_imaging.training.dataset import ZemaxBurstDataset, DEFAULT_EXCLUDED_IDS


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'runs/brightness_seed_repeats_20261012';out.mkdir(exist_ok=True)
    policy={'seeds':[5400,5401,5402],'epochs_per_group':20,
            'gate':'Every seed: C2_last PSNR > C0_last, centered_low_mse < C0_last, SSIM loss <= 0.001',
            'selection':'Highest validation PSNR among C2 best checkpoints; ties choose smaller seed',
            'test_references':['original_B','same_seed_C0_best'],
            'note':'Repeated fine-tuning from one fixed B model; not three independent A-to-B retrainings'}
    if (out/'policy.json').exists() and json.loads((out/'policy.json').read_text())!=policy:
        raise ValueError('Policy changed')
    write_json(out/'policy.json',policy)
    write_json(out/'status.json',{'status':'running','pid':os.getpid()})
    directories={5400:root/'runs/brightness_comparison_20261011'}
    for seed in (5401,5402):
        folder=out/f'seed_{seed}';directories[seed]=folder
        if not (folder/'status.json').exists() or json.loads((folder/'status.json').read_text())['status']!='completed':
            print(f'Starting seed {seed}: C0/C2 each 20 epochs',flush=True)
            subprocess.run([sys.executable,'-u',str(root/'scripts/run_brightness_comparison.py'),
                            '--output',str(folder),'--seed',str(seed),'--groups','C0','C2'],cwd=root,check=True)
    rows=[]
    for seed,folder in directories.items():
        data=json.loads((folder/'comparison.json').read_text())['metrics']
        a,b=data['C0_last'],data['C2_last']
        rows.append({'seed':seed,'delta_psnr_db':b['psnr_db']-a['psnr_db'],
                     'delta_ssim':b['ssim']-a['ssim'],
                     'delta_centered_low_mse':b['centered_low_mse']-a['centered_low_mse'],
                     'c2_best_psnr_db':data['C2_best']['psnr_db']})
    passed=all(r['delta_psnr_db']>0 and r['delta_centered_low_mse']<0 and r['delta_ssim']>=-0.001 for r in rows)
    result={'seeds':rows,'gate_passed':passed,
            'mean_delta_psnr_db':float(np.mean([r['delta_psnr_db'] for r in rows]))}
    write_json(out/'validation_stability.json',result)
    if not passed:
        write_json(out/'status.json',{'status':'completed_validation_gate_failed','test_evaluated':False})
        print('Validation stability gate failed; test set remains untouched',flush=True)
        return
    selected=max(rows,key=lambda r:(r['c2_best_psnr_db'],-r['seed']))['seed']
    paths={'C2':directories[selected]/'C2/best.pt',
           'original_B':root/'runs/train_b_expanded_split_residual_ssim_w020_stage2/best.pt',
           'C0':directories[selected]/'C0/best.pt'}
    selection={'seed':selected,'checkpoints':{k:{'path':str(p),'sha256':sha256_file(p)} for k,p in paths.items()}}
    lock=out/'locked_selection.json'
    if lock.exists() and json.loads(lock.read_text())!=selection:raise ValueError('Locked selection changed')
    write_json(lock,selection)
    if not (out/'test_summary.json').exists():
        torch.set_num_threads(4)
        ds=ZemaxBurstDataset(root/'runs/div2k_train_v1','b','test',exclude_ids=DEFAULT_EXCLUDED_IDS)
        if len(ds)!=80:raise ValueError('Test count changed')
        loader=DataLoader(ds,batch_size=8,shuffle=False)
        evaluated={}
        for name,path in paths.items():
            model,_=load_model(path,'b',torch.device('cpu'));evaluated[name]=evaluate(model,loader)
            with (out/f'test_{name}.csv').open('w',newline='',encoding='utf-8') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(evaluated[name][0]));writer.writeheader();writer.writerows(evaluated[name])
        report={'split':'test','samples':80,'locked_selection':selection,
                'metrics':{k:averages(v) for k,v in evaluated.items()},'paired':{}}
        for reference in ('original_B','C0'):
            report['paired'][reference]={}
            assert [r['sample_id'] for r in evaluated['C2']]==[r['sample_id'] for r in evaluated[reference]]
            for metric in report['metrics']['C2']:
                values=np.array([a[metric]-b[metric] for a,b in zip(evaluated['C2'],evaluated[reference])])
                report['paired'][reference][metric]={'mean':float(values.mean()),'ci95_mean':confidence_interval(values)}
        write_json(out/'test_summary.json',report)
    write_json(out/'status.json',{'status':'completed','test_evaluated':True,'selected_seed':selected})
    print('Repeat validation and locked test evaluation completed',flush=True)


if __name__=='__main__': main()
