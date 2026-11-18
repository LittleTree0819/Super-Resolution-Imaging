"""Resumable B-stage initial learning-rate comparison; validation only."""
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
from run_brightness_comparison import evaluate, averages, write_json
from super_resolution_imaging.training.cli import run_epoch, seed_everything, save_checkpoint, sha256_file
from super_resolution_imaging.training.dataset import ZemaxBurstDataset, DEFAULT_EXCLUDED_IDS
from super_resolution_imaging.training.evaluation import load_model, confidence_interval


GROUPS={'R0':1e-4,'R1':5e-5,'R2':2.5e-5}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--scheduler-study',action='store_true',help='Compare plateau and cosine at initial lr=1e-4')
    parser.add_argument('--weight-decay-study',action='store_true',help='Compare AdamW decay 1e-5, 0 and 1e-4 at lr=1e-4 with plateau')
    args=parser.parse_args()
    if args.scheduler_study and args.weight_decay_study:
        parser.error('Select only one study type')
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1]
    source=root/'runs/train_a_expanded_split_residual_ssim_w020_stage1/best.pt'
    dataset=root/'runs/div2k_train_v1'
    expected='86afbf8a198b8c9565c0aff39b856751301d30242baf8f13611e262f9684a86e'
    if sha256_file(source)!=expected: raise ValueError('A initialization checkpoint changed')
    epochs=1 if args.smoke else 20
    groups={'R0':1e-4,'R1':1e-4} if args.scheduler_study else GROUPS
    if args.weight_decay_study: groups={'R0':1e-4,'R1':1e-4,'R2':1e-4}
    decays={'R0':1e-5,'R1':0.,'R2':1e-4} if args.weight_decay_study else {k:1e-5 for k in groups}
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    specification={'groups':groups,'epochs':epochs,'seed':5400,'batch_size':8,'threads':4,
        'smoke':args.smoke,'source':str(source),'source_sha256':expected,
        'dataset_manifest_sha256':sha256_file(dataset/'manifest.jsonl'),
        'quality_policy_sha256':sha256_file(dataset/'quality_exclusions.json'),
        'optimizer':'AdamW','weight_decay':1e-5,'ssim_weight':.2,'component_weight':.2,
        'global_brightness_weight':0.,'local_low_frequency_weight':0.,
        'scheduler':{'kind':'ReduceLROnPlateau','monitor':'validation_l1','patience':5,'factor':.5,'min_lr':1e-6},
        'early_stopping':False,'test_evaluation':False,
        'selection_rule':'Highest epoch20 validation PSNR; ties use SSIM then lower L1; retain best within selected group as candidate'}
    if args.scheduler_study:
        specification['scheduler_study']=True
        specification['group_schedulers']={'R0':specification['scheduler'],
            'R1':{'kind':'CosineAnnealingLR','T_max':epochs,'eta_min':1e-6,'step':'after_each_epoch'}}
    if args.weight_decay_study:
        specification['weight_decay_study']=True
        specification['group_weight_decays']=decays
    if (out/'experiment.json').exists() and json.loads((out/'experiment.json').read_text())!=specification:
        raise ValueError('Experiment specification changed')
    write_json(out/'experiment.json',specification)
    train=ZemaxBurstDataset(dataset,'b','train',limit=2 if args.smoke else None,exclude_ids=DEFAULT_EXCLUDED_IDS)
    validation=ZemaxBurstDataset(dataset,'b','validation',limit=2 if args.smoke else None,exclude_ids=DEFAULT_EXCLUDED_IDS)
    if not args.smoke and (len(train)!=2586 or len(validation)!=72):raise ValueError('Dataset counts changed')
    ids={label:[r['sample_id'] for r in data.records] for label,data in [('train',train),('validation',validation)]}
    val_loader=DataLoader(validation,batch_size=8,shuffle=False)
    status={'status':'running','pid':os.getpid(),'started_at':time.strftime('%Y-%m-%d %H:%M:%S'),'groups':{}}
    write_json(out/'status.json',status)
    try:
        for name,lr in groups.items():
            folder=out/name;folder.mkdir(exist_ok=True)
            seed_everything(5400)
            model,original=load_model(source,'a',torch.device('cpu'))
            optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=decays[name])
            cosine=args.scheduler_study and name=='R1'
            scheduler=(torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-6)
                       if cosine else torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=5,factor=.5,min_lr=1e-6))
            generator=torch.Generator().manual_seed(5400)
            loader=DataLoader(train,batch_size=8,shuffle=True,generator=generator)
            cfg=dict(original['configuration'],group='b',initial_learning_rate=lr,epochs=epochs,
                weight_decay=decays[name],
                global_brightness_weight=0.,local_low_frequency_weight=0.,seed=5400,
                initialized_from={'path':str(source),'sha256':expected,'source_group':'a'},
                scheduler=(specification['group_schedulers'][name] if args.scheduler_study else specification['scheduler']),early_stopping=False)
            history=[];start=1
            if (folder/'last.pt').exists():
                saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
                if saved['configuration']!=cfg or saved['sample_ids']!=ids:raise ValueError('Resume mismatch')
                model.load_state_dict(saved['model_state']);optimizer.load_state_dict(saved['optimizer_state'])
                scheduler.load_state_dict(saved['scheduler_state']);generator.set_state(saved['generator_state'])
                torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
                history=saved['history'];start=saved['epoch']+1
                best_epoch=saved['best_epoch'];best_psnr=saved['best_validation_psnr_db']
            else:
                baseline=averages(evaluate(model,val_loader));best_epoch=0;best_psnr=baseline['psnr_db']
                write_json(folder/'epoch_zero.json',baseline)
                save_checkpoint(folder/'best.pt',{'model_state':model.state_dict(),'configuration':cfg,
                    'epoch':0,'best_epoch':0,'best_validation_psnr_db':best_psnr})
            status['current_group']=name;status['next_epoch']=start;write_json(out/'status.json',status)
            print(f'{name}: initial_lr={lr:.2e}, weight_decay={decays[name]:.2e}, starting epoch {start}/{epochs}',flush=True)
            for epoch in range(start,epochs+1):
                begin=time.monotonic();used_lr=optimizer.param_groups[0]['lr']
                train_l1,train_psnr=run_epoch(model,loader,torch.nn.L1Loss(),torch.device('cpu'),optimizer,.2,.2)
                validation_metrics=averages(evaluate(model,val_loader))
                if cosine: scheduler.step()
                else: scheduler.step(validation_metrics['l1'])
                history.append({'epoch':epoch,'used_learning_rate':used_lr,'next_learning_rate':optimizer.param_groups[0]['lr'],
                    'train_l1':train_l1,'train_psnr_db':train_psnr,'seconds':time.monotonic()-begin,**validation_metrics})
                improved=validation_metrics['psnr_db']>best_psnr
                if improved:best_psnr=validation_metrics['psnr_db'];best_epoch=epoch
                state={'epoch':epoch,'best_epoch':best_epoch,'best_validation_psnr_db':best_psnr,
                    'configuration':cfg,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                    'scheduler_state':scheduler.state_dict(),'history':history,'sample_ids':ids,
                    'generator_state':generator.get_state(),'torch_rng':torch.get_rng_state(),
                    'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
                save_checkpoint(folder/'last.pt',state)
                if improved:save_checkpoint(folder/'best.pt',state)
                write_json(folder/'history.json',history)
                status['groups'][name]={'completed_epoch':epoch,'best_epoch':best_epoch,'validation':validation_metrics}
                status['next_epoch']=epoch+1;write_json(out/'status.json',status)
                print(f"{name} epoch={epoch}/{epochs} lr={used_lr:.2e} next_lr={optimizer.param_groups[0]['lr']:.2e} seconds={history[-1]['seconds']:.1f} val_psnr={validation_metrics['psnr_db']:.4f} val_ssim={validation_metrics['ssim']:.5f}",flush=True)
            status['groups'][name]={'completed_epoch':epochs,'best_epoch':best_epoch,'validation':{k:history[-1][k] for k in validation_metrics} if start==1 else averages(evaluate(model,val_loader))}
            write_json(out/'status.json',status)
            write_json(folder/'summary.json',{'completed_epoch':epochs,'best_epoch':best_epoch,'initial_learning_rate':lr,'best_validation_psnr_db':best_psnr})
        evaluated={}
        for name in groups:
            for kind in ('last','best'):
                model,_=load_model(out/name/f'{kind}.pt','b',torch.device('cpu'))
                evaluated[f'{name}_{kind}']=evaluate(model,val_loader)
        summary={'split':'validation','samples':len(validation),'metrics':{k:averages(v) for k,v in evaluated.items()},'paired':{}}
        for name in list(groups)[1:]:
            summary['paired'][name]={}
            for metric in summary['metrics']['R0_last']:
                assert [r['sample_id'] for r in evaluated[f'{name}_last']]==[r['sample_id'] for r in evaluated['R0_last']]
                delta=np.array([a[metric]-b[metric] for a,b in zip(evaluated[f'{name}_last'],evaluated['R0_last'])])
                summary['paired'][name][metric]={'mean':float(delta.mean()),'ci95_mean':confidence_interval(delta)}
        chosen=max(groups,key=lambda n:(summary['metrics'][f'{n}_last']['psnr_db'],summary['metrics'][f'{n}_last']['ssim'],-summary['metrics'][f'{n}_last']['l1']))
        summary['selection']={'group':chosen,'initial_learning_rate':groups[chosen],
            'weight_decay':decays[chosen],
            'rule':specification['selection_rule'],'candidate_checkpoint':str(out/chosen/'best.pt'),
            'candidate_sha256':sha256_file(out/chosen/'best.pt'),
            'limitation':'One seed and 20-epoch budget; not a globally optimal learning rate'}
        for name,rows in evaluated.items():
            with (out/f'validation_{name}.csv').open('w',newline='',encoding='utf-8') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
        if args.scheduler_study: summary['scheduler_labels']={'R0':'plateau','R1':'cosine'}
        if args.weight_decay_study: summary['weight_decay_labels']=decays
        write_json(out/'comparison.json',summary)
        title=('# Stage B Weight Decay Comparison' if args.weight_decay_study else
               '# Stage B Scheduler Comparison' if args.scheduler_study else '# Stage B Initial Learning Rate Comparison')
        lines=[title,'', '|Model|L1|PSNR (dB)|SSIM|Brightness bias|Local low-frequency MSE|','|---|---:|---:|---:|---:|---:|']
        for name,m in summary['metrics'].items():lines.append(f"|{name}|{m['l1']:.6f}|{m['psnr_db']:.4f}|{m['ssim']:.6f}|{m['abs_bias']:.6f}|{m['centered_low_mse']:.6f}|")
        lines+=['',f'Fixed-budget selection: {chosen} with an initial learning rate of {groups[chosen]:.2e}. Results cover one seed and 20 epochs on validation.']
        if args.weight_decay_study:lines.append(f'Selected weight decay = {decays[chosen]:.2e}.')
        (out/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        status.update(status='completed',selection=summary['selection'],finished_at=time.strftime('%Y-%m-%d %H:%M:%S'))
        write_json(out/'status.json',status);print('All learning-rate groups and validation comparison completed',flush=True)
    except BaseException as exc:
        status.update(status='failed',error=repr(exc));write_json(out/'status.json',status);raise


if __name__=='__main__':main()
