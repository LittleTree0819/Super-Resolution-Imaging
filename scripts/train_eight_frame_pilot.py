"""Matched four/new-four/eight/repeated-eight pilot, without full-data warm starts."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from torch import nn
from train_symmetric_pilot_comparison import PilotDataset, make_model, train_stage, write_json
from super_resolution_imaging.training.cli import seed_everything, sha256_file
from super_resolution_imaging.training.evaluation import confidence_interval

VARIANTS={'four':(0,1,2,3),'new_four':(4,5,6,7),'eight':tuple(range(8)),
          'repeat_eight':(0,1,2,3,0,1,2,3)}


def expand_eight(state):
    expanded={k:v.clone() for k,v in state.items()}
    expanded['enc1.0.weight']=state['enc1.0.weight'].repeat(1,2,1,1)/2
    return expanded


def smoke():
    seed_everything(5400)
    four=make_model(4);eight=make_model(8)
    for head in (four.low_head[0],four.high_head[0]):nn.init.normal_(head.weight,0,.001)
    eight.load_state_dict(expand_eight(four.state_dict()))
    x=torch.rand(2,4,32,32)
    error=float((four(x)-eight(x.repeat(1,2,1,1))).abs().max().detach())
    assert error<1e-5
    eight(torch.rand(2,8,32,32)).mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in eight.parameters())
    return {'function_matched_warm_start_max_error':error,'eight_channel_backward':'passed'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('runs/eight_frame_quarter_network_pilot'))
    parser.add_argument('--smoke-only',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4)
    check=smoke()
    if args.smoke_only:print(json.dumps(check));return
    root=Path(__file__).resolve().parents[1]
    dataset=root/'runs/eight_frame_quarter_pilot_32'
    if not (dataset/'COMPLETE.json').exists():raise ValueError('Dataset incomplete')
    records=[json.loads(s) for s in (dataset/'manifest.jsonl').read_text().splitlines()]
    selection=json.loads((root/'data/nine_frame_pilot_sources/selection.json').read_text())['samples']
    expected={Path(r['pilot_source']).stem+'_00':r for r in selection}
    if len(records)!=32 or {r['sample_id'] for r in records}!=set(expected):raise ValueError('Scene set changed')
    for record in records:
        folder=dataset/'samples'/record['sample_id']
        if sha256_file(folder/'source.png')!=expected[record['sample_id']]['source_png_sha256']:raise ValueError('Source hash mismatch')
        if sha256_file(folder/'target.png')!=record['target_sha256']:raise ValueError('Target hash mismatch')
        for group in ('a','b'):
            if len(record['output_sha256'][group])!=8:raise ValueError('Frame count mismatch')
            for i,digest in enumerate(record['output_sha256'][group]):
                if sha256_file(folder/group/f'frame_{i:02d}.png')!=digest:raise ValueError('Frame hash mismatch')
    ordered=sorted(selection,key=lambda r:r['rank'])
    validation=[Path(r['pilot_source']).stem+'_00' for i,r in enumerate(ordered) if i%4==2]
    train=sorted(set(expected)-set(validation))
    frozen=json.loads((root/'runs/symmetric_pilot_experiment_plan.json').read_text())
    if set(train)!=set(frozen['train_ids']) or validation!=frozen['validation_ids']:raise ValueError('Original pilot split changed')
    if {s.split('_')[0] for s in train}&{s.split('_')[0] for s in validation}:raise ValueError('Source leakage')
    protocol={'dataset':str(dataset),'manifest_sha256':sha256_file(dataset/'manifest.jsonl'),
        'config_sha256':sha256_file(dataset/'dataset.json'),'train_ids':train,'validation_ids':validation,
        'seeds':[5400,5401,5402],'a_epochs':20,'b_epochs':60,'batch_size':8,'patch_lr':64,
        'base_channels':24,'a_lr':3e-4,'b_lr':1e-4,'weight_decay':1e-5,
        'scheduler':'val_full_l1 plateau patience5 factor0.5 min_lr1e-6','early_stopping':False,
        'ssim_weight':.2,'component_weight':.2,'global_brightness_weight':0.,'local_low_frequency_weight':0.,
        'variants':{k:list(v) for k,v in VARIANTS.items()},'primary':'fixed final B epoch interior crop16',
        'common_A':'from scratch on pilot train sources only; final A weights shared across B variants',
        'target_reference':'Unchanged centered GT; new_four first-frame base is shifted and must learn the fixed mapping',
        'eight_initialization':'duplicate first-convolution weights at half strength; function matched for repeated input',
        'no_formal_test':True}
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Protocol changed')
    write_json(out/'protocol.json',protocol);write_json(out/'smoke.json',check)
    status={'status':'running','pid':os.getpid(),'completed_stages':[]}
    write_json(out/'status.json',status)
    try:
        for seed in protocol['seeds']:
            seed_everything(seed);common=make_model(4)
            status.update(seed=seed,stage='a');write_json(out/'status.json',status)
            center_train=PilotDataset(dataset,train,(0,1,2,3),64,True)
            center_val=PilotDataset(dataset,validation,(0,1,2,3),center=True)
            shared=train_stage(common,center_train,center_val,out/f'seed{seed}_a',20,3e-4,seed,
                {'protocol':protocol,'seed':seed,'variant':'a'},plateau=True)
            status['completed_stages'].append(f'{seed}_a')
            for name,indices in VARIANTS.items():
                status.update(stage=name);write_json(out/'status.json',status)
                seed_everything(seed);model=make_model(len(indices))
                model.load_state_dict(shared if len(indices)==4 else expand_eight(shared))
                train_data=PilotDataset(dataset,train,indices,64)
                val_data=PilotDataset(dataset,validation,indices)
                train_stage(model,train_data,val_data,out/f'seed{seed}_{name}',60,1e-4,seed,
                    {'protocol':protocol,'seed':seed,'variant':name},plateau=True)
                status['completed_stages'].append(f'{seed}_{name}')
                write_json(out/'status.json',status)
        results={kind:{name:{str(seed):json.loads((out/f'seed{seed}_{name}'/f'{kind}_evaluation.json').read_text())
                    for seed in protocol['seeds']} for name in VARIANTS} for kind in ('last','best')}
        gains={}
        for a,b in [('eight','four'),('eight','new_four'),('eight','repeat_eight'),('new_four','four')]:
            differences=[results['last'][a][str(seed)]['metrics']['16']['psnr']-
                         results['last'][b][str(seed)]['metrics']['16']['psnr'] for seed in protocol['seeds']]
            ssim=[results['last'][a][str(seed)]['metrics']['16']['ssim']-
                  results['last'][b][str(seed)]['metrics']['16']['ssim'] for seed in protocol['seeds']]
            gains[f'{a}_minus_{b}']={'mean_psnr_db':float(np.mean(differences)),'per_seed_psnr_db':differences,
                'positive_seeds':sum(v>0 for v in differences),'mean_ssim_delta':float(np.mean(ssim))}
        # Average seeds within each validation source before bootstrapping sources.
        paired={}
        for control in ('four','new_four','repeat_eight'):
            per_source=[]
            for sid in validation:
                deltas=[]
                for seed in protocol['seeds']:
                    def lookup(name):return next(r for r in results['last'][name][str(seed)]['per_sample'] if r['sample_id']==sid and r['margin_hr']==16)['psnr']
                    deltas.append(lookup('eight')-lookup(control))
                per_source.append(float(np.mean(deltas)))
            paired[control]={'seed_averaged_source_delta_db':per_source,'ci95_mean':confidence_interval(np.array(per_source)),
                             'note':'Eight sources only; seed averaging does not create 24 independent samples'}
        report={'protocol':protocol,'results':results,'paired_gains':gains,'paired_source_bootstrap':paired,
                'limitations':['Only eight exploratory validation sources','Four/eight channels differ in first-layer parameter count; repeat_eight is the matched-capacity control',
                               'New-four uses a different bicubic reference frame; centered GT unchanged',
                               'Low-information sources remain in the primary comparison; no post-hoc removal']}
        write_json(out/'summary.json',report)
        lines=['# Eight-Frame Network Pilot with Additional Sampling Phases','', '24 training and 8 validation scenes across three seeds. The primary comparison uses Stage B epoch 60 and removes 16 HR pixels per border.','',
            '|Variant|Full-image PSNR|Interior PSNR|Full-image SSIM|Interior SSIM|','|---|---:|---:|---:|---:|']
        for name in VARIANTS:
            scores=[results['last'][name][str(s)]['metrics'] for s in protocol['seeds']]
            lines.append('|'+name+'|'+'|'.join(f'{np.mean([r[m][k] for r in scores]):.6f}' for m,k in [('0','psnr'),('16','psnr'),('0','ssim'),('16','ssim')])+'|')
        for comparison,gain in gains.items():lines.append(f'\n{comparison}: {gain["mean_psnr_db"]:+.6f} dB. By seed: {gain["per_seed_psnr_db"]}.')
        lines.append('\nExploratory validation results. See summary.json for per-image, per-seed and best-checkpoint results.')
        (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        status.update(status='completed');write_json(out/'status.json',status)
        print('All four variants and three seeds completed',flush=True)
    except BaseException as exc:
        status.update(status='failed',error=repr(exc));write_json(out/'status.json',status);raise


if __name__=='__main__':main()
