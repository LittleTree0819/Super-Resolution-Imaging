"""Complete matched-update four/new-four controls using frozen shared A weights."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from run_eight_frame_update_checks import train_updates, evaluate
from train_symmetric_pilot_comparison import PilotDataset, make_model, write_json
from super_resolution_imaging.training.cli import seed_everything, sha256_file
from super_resolution_imaging.training.evaluation import confidence_interval

VARIANTS={'four':(0,1,2,3),'new_four':(4,5,6,7)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1]
    previous=root/'runs/eight_frame_update_checks'
    protocol=json.loads((previous/'protocol.json').read_text())
    if json.loads((previous/'status.json').read_text())['status']!='completed':raise ValueError('Previous comparison not complete')
    data=Path(protocol['dataset'])
    if sha256_file(data/'manifest.jsonl')!=protocol['manifest_sha256']:raise ValueError('Dataset changed')
    source_paths={str(seed):previous/f'seed{seed}_a/last.pt' for seed in protocol['followup_seeds']}
    plan={'parent_protocol':protocol,'new_variants':{k:list(v) for k,v in VARIANTS.items()},
          'shared_A_sha256':{seed:sha256_file(path) for seed,path in source_paths.items()},
          'steps':1 if args.smoke else protocol['followup_b_updates'],'smoke':args.smoke,
          'primary':'fixed update2000 interior crop16; compare eight against each four-frame group',
          'new_four_reference':'first input is shifted; GT remains the original centered image',
          'no_formal_test':True}
    out=root/('runs/eight_frame_four_controls_smoke' if args.smoke else 'runs/eight_frame_four_controls')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=plan:raise ValueError('Protocol changed')
    write_json(out/'protocol.json',plan)
    status={'status':'running','pid':os.getpid(),'completed_stages':[]};write_json(out/'status.json',status)
    seeds=[5400] if args.smoke else protocol['followup_seeds']
    try:
        for seed in seeds:
            shared=torch.load(source_paths[str(seed)],weights_only=False,map_location='cpu')
            if shared['update']!=1000:raise ValueError('Unexpected A update count')
            for name,indices in VARIANTS.items():
                seed_everything(seed);model=make_model(4);model.load_state_dict(shared['model_state'])
                train=PilotDataset(data,protocol['train_ids'],indices)
                val=PilotDataset(data,protocol['validation_ids'],indices)
                status.update(seed=seed,variant=name);write_json(out/'status.json',status)
                train_updates(model,train,val,out/f'seed{seed}_{name}',plan['steps'],protocol['b_lr'],seed,protocol['followup_batch'],
                    {'protocol':protocol,'supplement':plan,'variant':name,'seed':seed},every=1 if args.smoke else 100)
                status['completed_stages'].append(f'{seed}_{name}');write_json(out/'status.json',status)
        if args.smoke:
            status.update(status='completed_smoke');write_json(out/'status.json',status);return
        results={}
        for kind in ('last','best'):
            results[kind]={}
            for name in ('four','new_four','eight','repeat_eight'):
                folder=out if name in VARIANTS else previous
                results[kind][name]={str(seed):json.loads((folder/f'seed{seed}_{name}/{kind}_evaluation.json').read_text()) for seed in seeds}
        means={kind:{name:{margin:{k:float(np.mean([r['metrics'][margin][k] for r in values.values()]))
                    for k in ('psnr_db','ssim','l1','hf_rmse')} for margin in ('0','16')}
                    for name,values in variants.items()} for kind,variants in results.items()}
        paired={}
        for control in ('four','new_four','repeat_eight'):
            deltas=[results['last']['eight'][str(seed)]['metrics']['16']['psnr_db']-
                    results['last'][control][str(seed)]['metrics']['16']['psnr_db'] for seed in seeds]
            sources={}
            for metric in ('psnr_db','ssim','l1','hf_rmse'):
                source_deltas=[]
                for sid in protocol['validation_ids']:
                    differences=[]
                    for seed in seeds:
                        def value(name):return next(r[metric] for r in results['last'][name][str(seed)]['per_sample'] if r['sample_id']==sid and r['margin']==16)
                        differences.append(value('eight')-value(control))
                    source_deltas.append(float(np.mean(differences)))
                sources[metric]={'mean':float(np.mean(source_deltas)),'ci95_mean':confidence_interval(np.array(source_deltas)),
                    'per_source_deltas':source_deltas}
            paired[f'eight_minus_{control}']={'per_seed_psnr_db':deltas,'mean_psnr_db':float(np.mean(deltas)),
                'positive_seeds':sum(v>0 for v in deltas),'source_averaged_paired':sources}
        best_four=max(VARIANTS,key=lambda name:means['last'][name]['16']['psnr_db'])
        summary={'results':results,'mean_metrics':means,'paired':paired,'best_of_two_four_frame_variants':best_four,
            'limitations':['Only 8 exploratory validation sources; no held-out test evaluation',
                'Best four refers only to the two tested layouts, selected on this same validation set',
                'Four/eight models have different first-layer capacity; repeat_eight is the capacity control',
                'new_four has a shifted bicubic base while centered GT is unchanged',
                'Full-image training and matched update budget differ from the earlier patch pilot']}
        write_json(out/'summary.json',summary)
        lines=['# Four Input Variants at Matched Update Counts','', '24 training and 8 validation scenes across three seeds. The primary comparison uses 2,000 Stage B updates and removes 16 HR pixels per border.','',
               '|Input|Full-image PSNR|Interior PSNR|Interior SSIM|Interior L1|Interior high-frequency RMSE|','|---|---:|---:|---:|---:|---:|']
        for name in ('four','new_four','eight','repeat_eight'):
            m=means['last'][name]
            lines.append(f"|{name}|{m['0']['psnr_db']:.6f}|{m['16']['psnr_db']:.6f}|{m['16']['ssim']:.6f}|{m['16']['l1']:.6f}|{m['16']['hf_rmse']:.6f}|")
        for name,pair in paired.items():lines.append(f'\n{name}: {pair["mean_psnr_db"]:+.6f} dB. By seed: {pair["per_seed_psnr_db"]}.')
        lines.append(f'\nHigher interior mean PSNR at update 2,000 among the two tested layouts: {best_four}.')
        (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        status.update(status='completed');write_json(out/'status.json',status)
        print('Supplementary four-frame controls and joint comparison completed',flush=True)
    except BaseException as exc:
        status.update(status='failed',error=repr(exc));write_json(out/'status.json',status);raise


if __name__=='__main__':main()
