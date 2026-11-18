"""Four-scene overfit gate, then matched-update eight/repeated-eight pilot."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from train_symmetric_pilot_comparison import PilotDataset, make_model, write_json
from train_eight_frame_pilot import expand_eight, smoke
from super_resolution_imaging.training.cli import run_epoch, seed_everything, save_checkpoint, sha256_file
from super_resolution_imaging.training.model import gaussian_blur
from super_resolution_imaging.training.baselines import bicubic, metrics
from super_resolution_imaging.training.evaluation import tensor_image, confidence_interval
from PIL import Image, ImageDraw

INDICES={'four':(0,1,2,3),'eight':tuple(range(8)),'repeat_eight':(0,1,2,3,0,1,2,3)}


def evaluate(model,data):
    model.eval();rows=[]
    with torch.inference_mode():
        for b in DataLoader(data,batch_size=4,shuffle=False):
            p,c=model.forward_with_components(b['input']);t=b['target']
            for margin in (0,16):
                pp,tt=(p,t) if margin==0 else (p[...,margin:-margin,margin:-margin],t[...,margin:-margin,margin:-margin])
                l1,psnr,ssim=metrics(pp,tt)
                # Filter before cropping to avoid creating artificial interior boundaries.
                error=(p-gaussian_blur(p,4))-(t-gaussian_blur(t,4))
                if margin:error=error[...,margin:-margin,margin:-margin]
                hf=error.square().mean((1,2,3)).sqrt()
                for i,sid in enumerate(b['sample_id']):
                    rows.append({'sample_id':sid,'margin':margin,'l1':float(l1[i]),'psnr_db':float(psnr[i]),
                                 'ssim':float(ssim[i]),'hf_rmse':float(hf[i])})
    means={str(m):{k:float(np.mean([r[k] for r in rows if r['margin']==m]))
                  for k in ('l1','psnr_db','ssim','hf_rmse')} for m in (0,16)}
    return {'metrics':means,'per_sample':rows}


class BicubicModel(torch.nn.Module):
    def forward_with_components(self,x):return bicubic(x[:,:1]),{}


def preview(model,data,path):
    canvas=Image.new('L',(5*256,len(data)*280),255);d=ImageDraw.Draw(canvas)
    model.eval()
    with torch.inference_mode():
        for row in range(len(data)):
            b=data[row];p,c=model.forward_with_components(b['input'][None]);t=b['target'][0]
            values=(bicubic(b['input'][None,:1])[0,0],p[0,0],t,(p[0,0]-t).abs()*4,
                    ((p[0,0]-gaussian_blur(p,4)[0,0])-(t-gaussian_blur(t[None,None],4)[0,0])).abs()*8)
            for col,(v,label) in enumerate(zip(values,('B0 bicubic','prediction','target','4x error','8x highfreq error'))):
                canvas.paste(tensor_image(v),(col*256,row*280+24))
                d.text((col*256+3,row*280+4),b['sample_id']+' '+label,fill=0)
    canvas.save(path)


def train_updates(model,train,val,folder,steps,lr,seed,batch_size,configuration,every=100):
    folder.mkdir(parents=True,exist_ok=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=5,factor=.5,min_lr=1e-6)
    history=[];completed=0;best=-float('inf')
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',weights_only=False,map_location='cpu')
        if state['configuration']!=configuration:raise ValueError('Checkpoint protocol mismatch')
        model.load_state_dict(state['model_state']);optimizer.load_state_dict(state['optimizer_state'])
        scheduler.load_state_dict(state['scheduler_state']);completed=state['update'];history=state['history'];best=state['best_psnr']
    elif not (folder/'initial.json').exists():
        write_json(folder/'initial.json',evaluate(model,val))
    batches=(len(train)+batch_size-1)//batch_size
    start_time=time.monotonic();last_time=start_time
    while completed<steps:
        cycle=completed//batches;skip=completed%batches
        seed_everything(seed+cycle*1009)
        loader=DataLoader(train,batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed+cycle*1009),num_workers=0)
        for index,batch in enumerate(loader):
            if index<skip:continue
            l1,psnr=run_epoch(model,[batch],torch.nn.L1Loss(),torch.device('cpu'),optimizer,.2,.2)
            completed+=1
            if completed%every==0 or completed==steps:
                score=evaluate(model,val);used_lr=optimizer.param_groups[0]['lr']
                scheduler.step(score['metrics']['0']['l1'])
                current=score['metrics']['16']['psnr_db'];improved=current>best;best=max(best,current)
                history.append({'update':completed,'last_batch_train_l1':l1,'last_batch_train_psnr':psnr,
                    'metrics':score['metrics'],'used_learning_rate':used_lr,'next_learning_rate':optimizer.param_groups[0]['lr'],
                    'seconds_since_previous_evaluation':time.monotonic()-last_time})
                state={'update':completed,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                    'scheduler_state':scheduler.state_dict(),'history':history,'best_psnr':best,'configuration':configuration}
                save_checkpoint(folder/'last.pt',state)
                if improved:save_checkpoint(folder/'best.pt',state)
                write_json(folder/'history.json',history);write_json(folder/'progress.json',history[-1])
                print(f"{folder.name} updates={completed}/{steps} PSNR={current:.3f} L1={score['metrics']['16']['l1']:.5f} HF={score['metrics']['16']['hf_rmse']:.5f} lr={used_lr:.2e}",flush=True)
                last_time=time.monotonic()
            if completed==steps:break
    for kind in ('last','best'):
        state=torch.load(folder/f'{kind}.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model_state'])
        write_json(folder/f'{kind}_evaluation.json',{'update':state['update'],**evaluate(model,val)})
        preview(model,val,folder/f'{kind}_preview.png')
    return torch.load(folder/'last.pt',map_location='cpu',weights_only=False)['model_state']


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1];data=root/'runs/eight_frame_quarter_pilot_32'
    previous=json.loads((root/'runs/eight_frame_quarter_network_pilot/protocol.json').read_text())
    if sha256_file(data/'manifest.jsonl')!=previous['manifest_sha256']:raise ValueError('Dataset changed')
    selection=json.loads((root/'data/nine_frame_pilot_sources/selection.json').read_text())['samples']
    ordered=[Path(r['pilot_source']).stem+'_00' for r in sorted(selection,key=lambda r:r['rank'])
             if Path(r['pilot_source']).stem+'_00' in previous['train_ids']]
    chosen=[ordered[i] for i in (3,9,15,21)]
    out=root/('runs/eight_frame_update_checks_smoke' if args.smoke else 'runs/eight_frame_update_checks')
    out.mkdir(parents=True,exist_ok=True)
    protocol={'dataset':str(data),'manifest_sha256':previous['manifest_sha256'],'overfit_ids':chosen,
        'overfit_a_updates':1 if args.smoke else 1000,'overfit_b_updates':2 if args.smoke else 2000,
        'full_images':True,'overfit_batch':4,'gate':{'mean_interior_psnr_min':28.,'mean_psnr_gain_over_bicubic_min':3.,
            'hf_rmse_ratio_max':.8,'required_variants':['four','eight','repeat_eight']},
        'followup_seeds':[5400] if args.smoke else [5400,5401,5402],
        'followup_a_updates':1 if args.smoke else 1000,'followup_b_updates':2 if args.smoke else 2000,
        'followup_batch':8,'train_ids':previous['train_ids'],'validation_ids':previous['validation_ids'],
        'a_lr':3e-4,'b_lr':1e-4,'scheduler':'plateau L1 at every100 updates patience5 factor.5 min1e-6',
        'loss':'L1+.2*(1-SSIM)+.2*component; no global/local additions','no_formal_test':True,
        'smoke':args.smoke,'sampling':'Fixed per-cycle seed; exact order reconstructible at resume; no random patches'}
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Protocol changed')
    write_json(out/'protocol.json',protocol);write_json(out/'warm_start_smoke.json',smoke())
    state={'status':'running','pid':os.getpid(),'stage':'overfit'};write_json(out/'status.json',state)
    every=1 if args.smoke else 100
    try:
        center=PilotDataset(data,chosen,(0,1,2,3),center=True)
        seed_everything(5400);model=make_model(4)
        shared=train_updates(model,center,center,out/'overfit_a',protocol['overfit_a_updates'],3e-4,5400,4,
            {'protocol':protocol,'variant':'overfit_a'},every)
        outcomes={}
        for name,indices in INDICES.items():
            state['current_variant']=name;write_json(out/'status.json',state)
            images=PilotDataset(data,chosen,indices)
            seed_everything(5400);model=make_model(len(indices));model.load_state_dict(shared if len(indices)==4 else expand_eight(shared))
            folder=out/f'overfit_{name}'
            train_updates(model,images,images,folder,protocol['overfit_b_updates'],1e-4,5400,4,
                {'protocol':protocol,'variant':name},every)
            baseline=evaluate(BicubicModel(),images)['metrics']['16']
            result=json.loads((folder/'last_evaluation.json').read_text())['metrics']['16']
            gain=result['psnr_db']-baseline['psnr_db'];ratio=result['hf_rmse']/baseline['hf_rmse']
            outcomes[name]={'baseline':baseline,'last':result,'psnr_gain_db':gain,'hf_rmse_ratio':ratio,
                'passed':result['psnr_db']>=28 and gain>=3 and ratio<=.8}
        gate={'outcomes':outcomes,'passed':all(v['passed'] for v in outcomes.values()),
              'note':'Training-set fit gate only; does not demonstrate generalization or new-phase benefit'}
        write_json(out/'overfit_gate.json',gate)
        if not gate['passed']:
            state.update(status='completed_overfit_gate_failed',stage='overfit_completed_followup_not_started')
            write_json(out/'status.json',state);print('Overfit gate not passed; stop for diagnosis',flush=True);return
        state.update(stage='fixed_update_pilot');write_json(out/'status.json',state)
        for seed in protocol['followup_seeds']:
            center_train=PilotDataset(data,previous['train_ids'],(0,1,2,3),center=True)
            center_val=PilotDataset(data,previous['validation_ids'],(0,1,2,3),center=True)
            seed_everything(seed);model=make_model(4)
            shared=train_updates(model,center_train,center_val,out/f'seed{seed}_a',protocol['followup_a_updates'],3e-4,seed,8,
                {'protocol':protocol,'variant':'a','seed':seed},every)
            for name in ('eight','repeat_eight'):
                state.update(seed=seed,current_variant=name);write_json(out/'status.json',state)
                indices=INDICES[name];seed_everything(seed);model=make_model(8);model.load_state_dict(expand_eight(shared))
                train=PilotDataset(data,previous['train_ids'],indices)
                val=PilotDataset(data,previous['validation_ids'],indices)
                train_updates(model,train,val,out/f'seed{seed}_{name}',protocol['followup_b_updates'],1e-4,seed,8,
                    {'protocol':protocol,'variant':name,'seed':seed},every)
        results={name:{str(seed):json.loads((out/f'seed{seed}_{name}/last_evaluation.json').read_text())
                 for seed in protocol['followup_seeds']} for name in ('eight','repeat_eight')}
        differences=[results['eight'][str(seed)]['metrics']['16']['psnr_db']-
                     results['repeat_eight'][str(seed)]['metrics']['16']['psnr_db'] for seed in protocol['followup_seeds']]
        per_source=[]
        for sid in previous['validation_ids']:
            values=[]
            for seed in protocol['followup_seeds']:
                def value(name):return next(r['psnr_db'] for r in results[name][str(seed)]['per_sample'] if r['sample_id']==sid and r['margin']==16)
                values.append(value('eight')-value('repeat_eight'))
            per_source.append(float(np.mean(values)))
        write_json(out/'summary.json',{'results':results,'per_seed_psnr_gain_db':differences,
            'mean_psnr_gain_db':float(np.mean(differences)),'source_bootstrap_ci95':confidence_interval(np.array(per_source)),
            'note':'Only eight exploratory validation scenes; no formal test evaluation'})
        state.update(status='completed');write_json(out/'status.json',state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));write_json(out/'status.json',state);raise


if __name__=='__main__':main()
