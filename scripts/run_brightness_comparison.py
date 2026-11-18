"""Resumable fixed-budget C0-C3 experiment, followed by validation-only comparison."""
import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from super_resolution_imaging.training.cli import run_epoch, seed_everything, save_checkpoint, sha256_file
from super_resolution_imaging.training.dataset import ZemaxBurstDataset, DEFAULT_EXCLUDED_IDS
from super_resolution_imaging.training.evaluation import load_model, confidence_interval
from super_resolution_imaging.training.baselines import metrics
from super_resolution_imaging.training.model import gaussian_blur


def write_json(path, value):
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2),encoding='utf-8')
    temporary.replace(path)


def evaluate(model, loader):
    rows=[]
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            pred=model(batch['input']); target=batch['target']; error=pred-target
            low=gaussian_blur(error,4)
            quantities=dict(zip(('l1','psnr_db','ssim'),metrics(pred,target)))
            quantities.update(abs_bias=error.mean((1,2,3)).abs(),
                              low_mse=low.square().mean((1,2,3)),
                              centered_low_mse=(low-low.mean((1,2,3),keepdim=True)).square().mean((1,2,3)),
                              high_mse=(error-low).square().mean((1,2,3)))
            for i,sid in enumerate(batch['sample_id']):
                rows.append({'sample_id':sid,**{k:float(v[i]) for k,v in quantities.items()}})
    return rows


def averages(rows):
    return {k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='sample_id'}


def compare(output, loader, initial, groups=('C0','C1','C2','C3')):
    all_rows={'initial':evaluate(initial,loader)}
    for name in groups:
        for kind in ('last','best'):
            model,_=load_model(output/name/f'{kind}.pt','b',torch.device('cpu'))
            all_rows[f'{name}_{kind}']=evaluate(model,loader)
    summary={'split':'validation','samples':len(all_rows['initial']),
             'primary_comparison':'epoch20_last_against_C0_last',
             'metrics':{k:averages(v) for k,v in all_rows.items()},'paired':{}}
    for name in groups:
        if name=='C0': continue
        for reference in ('C0_last','initial'):
            key=f'{name}_last_minus_{reference}'
            summary['paired'][key]={}
            for metric in summary['metrics']['initial']:
                values=np.array([r[metric]-s[metric] for r,s in zip(all_rows[f'{name}_last'],all_rows[reference])])
                summary['paired'][key][metric]={'mean':float(values.mean()),'ci95_mean':confidence_interval(values),
                    'better_samples':int(((values>0) if metric in ('psnr_db','ssim') else (values<0)).sum())}
    for name,rows in all_rows.items():
        with (output/f'validation_{name}.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_json(output/'comparison.json',summary)
    lines=['# C0–C3 Fixed-Budget Validation Comparison','',f"Fixed validation set of {summary['samples']} images. Epoch 20 is the primary comparison, with best checkpoints reported as a secondary reference. The test set was unused.",'',
           '|Model|L1|PSNR (dB)|SSIM|Absolute brightness bias|Mean-centered low-frequency MSE|', '|---|---:|---:|---:|---:|---:|']
    for name,m in summary['metrics'].items():
        lines.append(f"|{name}|{m['l1']:.6f}|{m['psnr_db']:.4f}|{m['ssim']:.6f}|{m['abs_bias']:.6f}|{m['centered_low_mse']:.6f}|")
    (output/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true',help='Isolated one-epoch two-sample orchestration test')
    parser.add_argument('--seed',type=int,default=5400)
    parser.add_argument('--groups',nargs='+',choices=('C0','C1','C2','C3'),default=['C0','C1','C2','C3'])
    args=parser.parse_args()
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1]
    spec=json.loads((root/'configs/brightness_low_frequency_experiment.json').read_text())
    if args.seed!=5400: spec['first_seed']=args.seed
    if args.groups!=['C0','C1','C2','C3']:
        spec['groups']=[g for g in spec['groups'] if g['name'] in args.groups]
    if 'C0' not in args.groups: raise ValueError('C0 control is required')
    source=root/spec['initial_checkpoint']; dataset=root/spec['dataset']
    if sha256_file(source)!=spec['initial_checkpoint_sha256']: raise ValueError('Initial checkpoint changed')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    configuration={**spec,'status':'running','smoke':args.smoke,'threads':4,
                   'manifest_sha256':sha256_file(dataset/'manifest.jsonl'),
                   'quality_policy_sha256':sha256_file(dataset/'quality_exclusions.json')}
    config_path=out/'experiment.json'
    if config_path.exists() and json.loads(config_path.read_text())!=configuration:
        raise ValueError('Existing experiment configuration differs')
    write_json(config_path,configuration)
    train=ZemaxBurstDataset(dataset,'b','train',limit=2 if args.smoke else None,exclude_ids=DEFAULT_EXCLUDED_IDS)
    validation=ZemaxBurstDataset(dataset,'b','validation',limit=2 if args.smoke else None,exclude_ids=DEFAULT_EXCLUDED_IDS)
    if not args.smoke and (len(train)!=2586 or len(validation)!=72): raise ValueError('Sample counts changed')
    val_loader=DataLoader(validation,batch_size=8,shuffle=False)
    epochs=1 if args.smoke else 20
    ids={s:[r['sample_id'] for r in d.records] for s,d in [('train',train),('validation',validation)]}
    status={'pid':os.getpid(),'status':'running','started_at':time.strftime('%Y-%m-%d %H:%M:%S'), 'groups':{}}
    write_json(out/'status.json',status)
    try:
        for group in spec['groups']:
            name=group['name']; folder=out/name;folder.mkdir(exist_ok=True)
            seed_everything(args.seed)
            model,source_checkpoint=load_model(source,'b',torch.device('cpu'))
            optimizer=torch.optim.AdamW(model.parameters(),lr=2.5e-5,weight_decay=1e-5)
            generator=torch.Generator().manual_seed(args.seed)
            loader=DataLoader(train,batch_size=8,shuffle=True,generator=generator)
            cfg=dict(source_checkpoint['configuration'],global_brightness_weight=group['global_weight'],
                     local_low_frequency_weight=group['local_weight'],initial_learning_rate=2.5e-5,
                     seed=args.seed,initialized_from={'path':str(source),'sha256':sha256_file(source)},
                     experiment='fixed_budget_brightness',epochs=epochs,scheduler=None,early_stopping=False)
            history=[]; start=1
            if (folder/'last.pt').exists():
                state=torch.load(folder/'last.pt',weights_only=False,map_location='cpu')
                if state['configuration']!=cfg or state['sample_ids']!=ids: raise ValueError('Resume state mismatch')
                model.load_state_dict(state['model_state']);optimizer.load_state_dict(state['optimizer_state'])
                generator.set_state(state['generator_state']);torch.set_rng_state(state['torch_rng'])
                random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
                history=state['history'];start=state['epoch']+1
                best_psnr=state['best_validation_psnr_db'];best_epoch=state['best_epoch']
            else:
                baseline=averages(evaluate(model,val_loader));best_psnr=baseline['psnr_db'];best_epoch=0
                save_checkpoint(folder/'best.pt',{'model_state':model.state_dict(),'configuration':cfg,
                    'epoch':0,'best_epoch':0,'best_validation_psnr_db':best_psnr})
                write_json(folder/'epoch_zero.json',baseline)
            print(f'{name}: starting epoch {start}/{epochs}; fixed lr=2.5e-5',flush=True)
            for epoch in range(start,epochs+1):
                began=time.monotonic()
                train_l1,train_psnr=run_epoch(model,loader,torch.nn.L1Loss(),torch.device('cpu'),optimizer,
                    .2,.2,group['global_weight'],group['local_weight'],4)
                validation_metrics=averages(evaluate(model,val_loader))
                history.append({'epoch':epoch,'train_l1':train_l1,'train_psnr_db':train_psnr,
                                'seconds':time.monotonic()-began,**validation_metrics})
                improved=validation_metrics['psnr_db']>best_psnr
                if improved: best_psnr=validation_metrics['psnr_db'];best_epoch=epoch
                state={'configuration':cfg,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                    'epoch':epoch,'best_epoch':best_epoch,'best_validation_psnr_db':best_psnr,'history':history,
                    'sample_ids':ids,'generator_state':generator.get_state(),'torch_rng':torch.get_rng_state(),
                    'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
                save_checkpoint(folder/'last.pt',state)
                if improved:save_checkpoint(folder/'best.pt',state)
                write_json(folder/'history.json',history)
                status['groups'][name]={'completed_epoch':epoch,'best_epoch':best_epoch,'validation':validation_metrics}
                status['current_group']=name;write_json(out/'status.json',status)
                print(f"{name} epoch={epoch}/{epochs} seconds={history[-1]['seconds']:.1f} val_psnr={validation_metrics['psnr_db']:.4f} val_ssim={validation_metrics['ssim']:.5f}",flush=True)
            status['groups'][name]={'completed_epoch':epochs,'best_epoch':best_epoch,
                'validation':{key:history[-1][key] for key in
                    ('l1','psnr_db','ssim','abs_bias','low_mse','centered_low_mse','high_mse')}}
            status['current_group']=name
            write_json(out/'status.json',status)
            write_json(folder/'summary.json',{'completed_epoch':epochs,'best_epoch':best_epoch,'best_validation_psnr_db':best_psnr})
        initial,_=load_model(source,'b',torch.device('cpu'))
        compare(out,val_loader,initial,[g['name'] for g in spec['groups']])
        status['status']='completed';status['finished_at']=time.strftime('%Y-%m-%d %H:%M:%S')
        write_json(out/'status.json',status)
        print('All groups and validation comparison completed',flush=True)
    except BaseException as exc:
        status.update(status='failed',error=repr(exc));write_json(out/'status.json',status)
        raise


if __name__=='__main__':main()
