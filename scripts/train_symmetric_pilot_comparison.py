"""Resumable paired A->B pilot: four real, nine real, nine repeated-four frames."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from super_resolution_imaging.training.model import BurstUNet2x
from super_resolution_imaging.training.cli import run_epoch, seed_everything, save_checkpoint
from super_resolution_imaging.training.baselines import metrics

FOUR=(0,1,7,8)
REPEAT=(0,1,8,7,8,1,8,7,8)
MAP=(0,1,3,2,3,1,3,2,3)


def write_json(path,data):
    tmp=path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8'); tmp.replace(path)


def image(path,size):
    with Image.open(path) as im:
        if im.mode!='L' or im.size!=(size,size): raise ValueError(f'Invalid image {path}')
        return torch.from_numpy(np.asarray(im,dtype=np.float32).copy()/255)


def verify_dataset(root):
    if not (root/'COMPLETE.json').is_file(): raise ValueError('Generation not complete')
    records=[json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines()]
    if len(records)!=32 or len({r['sample_id'] for r in records})!=32: raise ValueError('Expected 32 unique scenes')
    selection=json.loads(Path('data/nine_frame_pilot_sources/selection.json').read_text())['samples']
    expected={Path(r['pilot_source']).stem+'_00':r for r in selection}
    if set(expected)!={r['sample_id'] for r in records}: raise ValueError('Scene selection changed')
    old_records={r['sample_id']:r for r in [json.loads(s) for s in Path('runs/nine_frame_pilot_32/manifest.jsonl').read_text().splitlines()]}
    for record in records:
        folder=root/'samples'/record['sample_id']
        if sha256((folder/'source.png').read_bytes()).hexdigest()!=expected[record['sample_id']]['source_png_sha256']: raise ValueError('Source crop changed')
        for group in ('a','b'):
            for i in range(9):
                path=folder/group/f'frame_{i:02d}.png'
                image(path,128)
                if sha256(path.read_bytes()).hexdigest()!=record['output_sha256'][group][i]: raise ValueError(f'Hash mismatch: {path}')
        image(folder/'target.png',256)
        if sha256((folder/'target.png').read_bytes()).hexdigest()!=record['target_sha256']: raise ValueError('Target hash mismatch')
        if record['target_sha256']!=old_records[record['sample_id']]['target_sha256']: raise ValueError('Center GT changed between pilots')
    return records,selection


class PilotDataset(Dataset):
    def __init__(self,root,ids,indices,patch=None,center=False):
        self.items=[]; self.patch=patch
        for sid in ids:
            folder=root/'samples'/sid
            frames=torch.stack([image(folder/('a' if center else 'b')/f'frame_{i:02d}.png',128) for i in indices])
            self.items.append({'sample_id':sid,'input':frames,'target':image(folder/'target.png',256)[None]})
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        result=self.items[i].copy()
        if self.patch:
            y,x=torch.randint(129-self.patch,(2,)).tolist()
            result['input']=result['input'][:,y:y+self.patch,x:x+self.patch]
            result['target']=result['target'][:,2*y:2*(y+self.patch),2*x:2*(x+self.patch)]
        return result


def make_model(channels):
    return BurstUNet2x(24,split_residual_output=True,split_sigma=4.,input_channels=channels)


def expand_state(state):
    result={k:v.clone() for k,v in state.items()}
    weight=state['enc1.0.weight']; counts=[MAP.count(i) for i in range(4)]
    result['enc1.0.weight']=torch.stack([weight[:,i]/counts[i] for i in MAP],dim=1)
    return result


def evaluate(model,dataset):
    model.eval(); rows=[]
    with torch.inference_mode():
        for batch in DataLoader(dataset,batch_size=4,shuffle=False):
            pred=model(batch['input']); truth=batch['target']
            for margin in (0,16):
                p,t=(pred,truth) if not margin else (pred[:,:,margin:-margin,margin:-margin],truth[:,:,margin:-margin,margin:-margin])
                l1,psnr,ssim=metrics(p,t)
                rows.extend({'sample_id':sid,'margin_hr':margin,'l1':float(l1[i]),'psnr':float(psnr[i]),'ssim':float(ssim[i])} for i,sid in enumerate(batch['sample_id']))
    summary={str(m):{k:float(np.mean([r[k] for r in rows if r['margin_hr']==m])) for k in ('l1','psnr','ssim')} for m in (0,16)}
    return rows,summary


def train_stage(model,train,val,folder,epochs,lr,seed,configuration,plateau=False):
    folder.mkdir(parents=True,exist_ok=True)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-5)
    scheduler=(torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',patience=5,factor=.5,min_lr=1e-6)
               if plateau else None)
    history=[]; start=0; best=-float('inf')
    if (folder/'last.pt').exists():
        state=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        if state['configuration']!=configuration: raise ValueError('Resume protocol changed')
        model.load_state_dict(state['model_state']); optimizer.load_state_dict(state['optimizer_state'])
        if scheduler is not None: scheduler.load_state_dict(state['scheduler_state'])
        history=state['history']; start=state['epoch']; best=state['best_psnr']
    for epoch in range(start+1,epochs+1):
        # Identical patch coordinates/order across all B variants for a seed/epoch.
        seed_everything(seed+epoch*1009)
        loader=DataLoader(train,batch_size=8,shuffle=True,generator=torch.Generator().manual_seed(seed+epoch*1009),num_workers=0)
        began=time.perf_counter()
        l1,psnr=run_epoch(model,loader,nn.L1Loss(),torch.device('cpu'),optimizer,.2,.2)
        rows,score=evaluate(model,val)
        used_lr=optimizer.param_groups[0]['lr']
        if scheduler is not None: scheduler.step(score['0']['l1'])
        current=score['16']['psnr']; improved=current>best; best=max(best,current)
        history.append({'epoch':epoch,'train_l1':l1,'train_psnr':psnr,'validation':score,'seconds':time.perf_counter()-began,
                        'used_learning_rate':used_lr,'next_learning_rate':optimizer.param_groups[0]['lr']})
        state={'epoch':epoch,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'configuration':configuration,'history':history,'best_psnr':best}
        if scheduler is not None: state['scheduler_state']=scheduler.state_dict()
        save_checkpoint(folder/'last.pt',state)
        if improved: save_checkpoint(folder/'best.pt',state)
        write_json(folder/'history.json',history)
        print(f'{folder.name} epoch={epoch}/{epochs} train_l1={l1:.5f} val_full={score["0"]["psnr"]:.3f} val_inner={current:.3f} seconds={history[-1]["seconds"]:.1f}',flush=True)
    # Primary comparison uses fixed final epoch; best validation epoch is secondary.
    for which in ('last','best'):
        state=torch.load(folder/f'{which}.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model_state'])
        rows,score=evaluate(model,val)
        write_json(folder/f'{which}_evaluation.json',{'epoch':state['epoch'],'metrics':score,'per_sample':rows})
    return torch.load(folder/'last.pt',map_location='cpu',weights_only=False)['model_state']


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,default=Path('runs/nine_frame_pilot_32_symmetric'))
    p.add_argument('--output',type=Path,default=Path('runs/nine_frame_pilot_training'))
    p.add_argument('--a-epochs',type=int,default=20); p.add_argument('--b-epochs',type=int,default=60)
    p.add_argument('--seeds',nargs='+',type=int,default=[5400,5401,5402])
    p.add_argument('--smoke-only',action='store_true')
    args=p.parse_args(); torch.set_num_threads(4)
    if args.smoke_only:
        seed_everything(5400); m4=make_model(4); m9=make_model(9)
        # Nonzero heads ensure the warm-start test checks more than the bicubic base.
        for head in (m4.low_head[0],m4.high_head[0]): nn.init.normal_(head.weight,0,.001)
        m9.load_state_dict(expand_state(m4.state_dict()))
        x=torch.rand(2,4,32,32); a=m4(x); b=m9(x[:,MAP])
        error=float((a-b).abs().max().detach()); assert error<1e-5,error
        loss=(m9(torch.rand(2,9,32,32))-torch.rand(2,1,64,64)).abs().mean(); loss.backward()
        assert torch.isfinite(m9.enc1[0].weight.grad).all()
        print(json.dumps({'warm_start_max_error':error,'nine_channel_backward':'passed'})); return
    records,selection=verify_dataset(args.dataset)
    # Texture-stratified source split, frozen before training; all variants share it.
    ordered=sorted(selection,key=lambda r:r['rank'])
    val_ids=[Path(r['pilot_source']).stem+'_00' for i,r in enumerate(ordered) if i%4==2]
    train_ids=[r['sample_id'] for r in sorted(records,key=lambda r:r['sample_id']) if r['sample_id'] not in val_ids]
    if len(train_ids)!=24 or len(val_ids)!=8: raise ValueError('Unexpected split')
    if args.dataset.name=='nine_frame_pilot_32_symmetric':
        plan=json.loads(Path('runs/symmetric_pilot_experiment_plan.json').read_text())
        if set(plan['train_ids'])!=set(train_ids) or plan['validation_ids']!=val_ids:
            raise ValueError('Predeclared scene split changed')
        if (plan['a_epochs'],plan['b_epochs'],plan['seeds'])!=(args.a_epochs,args.b_epochs,args.seeds):
            raise ValueError('Predeclared epochs or seeds changed')
    sources=lambda ids:{sid.split('_')[0] for sid in ids}
    assert not sources(train_ids)&sources(val_ids)
    args.output.mkdir(parents=True,exist_ok=True)
    protocol={'dataset':str(args.dataset.resolve()),'manifest_sha256':sha256((args.dataset/'manifest.jsonl').read_bytes()).hexdigest(),'config_sha256':sha256((args.dataset/'dataset.json').read_bytes()).hexdigest(),'train_ids':train_ids,'validation_ids':val_ids,'seeds':args.seeds,'a_epochs':args.a_epochs,'b_epochs':args.b_epochs,'base_channels':24,'patch_lr':64,'batch_size':8,'a_lr':.0003,'b_lr':.0001,'component_loss_weight':.2,'ssim_loss_weight':.2,'sigma':4.,'primary':'fixed final B epoch, validation HR crop16','variants':{'four':FOUR,'nine':list(range(9)),'repeat9':REPEAT},'no_test_set':True}
    if (args.output/'protocol.json').exists() and json.loads((args.output/'protocol.json').read_text())!=json.loads(json.dumps(protocol)): raise ValueError('Frozen protocol changed')
    write_json(args.output/'protocol.json',protocol)
    for seed in args.seeds:
        seed_everything(seed); common=make_model(4)
        center_train=PilotDataset(args.dataset,train_ids,FOUR,64,True)
        center_val=PilotDataset(args.dataset,val_ids,FOUR,center=True)
        shared=train_stage(common,center_train,center_val,args.output/f'seed{seed}_a',args.a_epochs,.0003,seed,{'protocol':protocol,'seed':seed,'variant':'a'})
        for name,inds in [('four',FOUR),('nine',tuple(range(9))),('repeat9',REPEAT)]:
            seed_everything(seed); model=make_model(len(inds)); model.load_state_dict(shared if len(inds)==4 else expand_state(shared))
            train=PilotDataset(args.dataset,train_ids,inds,64); val=PilotDataset(args.dataset,val_ids,inds)
            train_stage(model,train,val,args.output/f'seed{seed}_{name}',args.b_epochs,.0001,seed,{'protocol':protocol,'seed':seed,'variant':name})
    results={}; gains={}
    for which in ('last','best'):
        results[which]={name:{str(seed):json.loads((args.output/f'seed{seed}_{name}'/f'{which}_evaluation.json').read_text()) for seed in args.seeds} for name in ('four','nine','repeat9')}
        gains[which]={}
        for margin in ('0','16'):
            gains[which][margin]={}
            for control in ('four','repeat9'):
                differences=[results[which]['nine'][str(seed)]['metrics'][margin]['psnr']-results[which][control][str(seed)]['metrics'][margin]['psnr'] for seed in args.seeds]
                ssim_differences=[results[which]['nine'][str(seed)]['metrics'][margin]['ssim']-results[which][control][str(seed)]['metrics'][margin]['ssim'] for seed in args.seeds]
                gains[which][margin][f'nine_minus_{control}']={'mean_db':float(np.mean(differences)),'per_seed_db':differences,'positive_seeds':sum(d>0 for d in differences),'mean_ssim_delta':float(np.mean(ssim_differences)),'per_seed_ssim_delta':ssim_differences}
    decision='Prioritize the existing expanded four-frame training; pilot has not established stable interior gains.'
    primary=gains['last']['16']
    if all(primary[f'nine_minus_{c}']['positive_seeds']==len(args.seeds) and primary[f'nine_minus_{c}']['mean_db']>=.05 and primary[f'nine_minus_{c}']['mean_ssim_delta']>=0 for c in ('four','repeat9')):
        decision='Nine-frame gains pass the exploratory consistency gate; consider an intermediate larger pilot before full nine-frame acquisition.'
    write_json(args.output/'summary.json',{'protocol':protocol,'results':results,'paired_psnr_gains':gains,'decision':decision,'limitations':['Only eight exploratory validation sources; no formal test set evaluation.','0.05 dB consistency gate is predeclared operational guidance, not statistical proof.','Fixed final-epoch comparison is primary; best-validation comparison is secondary.']})
    lines=['# Symmetric nine-frame exploratory comparison','', '24 training sources / 8 validation sources; three variants and matched random seeds.', '', 'Primary: fixed final B epoch, no best-checkpoint selection.','', '| Model | Full PSNR | Interior PSNR | Full SSIM | Interior SSIM |','|---|---:|---:|---:|---:|']
    for name in ('four','nine','repeat9'):
        vals=[results['last'][name][str(seed)]['metrics'] for seed in args.seeds]
        lines.append('| '+name+' | '+' | '.join(f'{np.mean([v[margin][metric] for v in vals]):.6f}' for margin,metric in [('0','psnr'),('16','psnr'),('0','ssim'),('16','ssim')])+' |')
    lines.extend(['','## Paired interior gains',''])
    for control in ('four','repeat9'):
        gain=primary[f'nine_minus_{control}']
        lines.append(f'Nine minus {control}: mean {gain["mean_db"]:+.6f} dB; per seed {gain["per_seed_db"]}; mean SSIM delta {gain["mean_ssim_delta"]:+.6f}.')
    lines.extend(['','## Next step','',decision,'','Exploratory only: eight validation sources, no formal held-out test evaluation. Poor convergence or differing learning curves would require extending this pilot before ruling out nine-frame learning. The 0.05 dB gate is operational guidance, not statistical significance.'])
    (args.output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(decision,flush=True)


if __name__=='__main__': main()
