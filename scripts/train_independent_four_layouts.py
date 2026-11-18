"""Independent 96/32-source replication of the two four-frame layouts."""
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from run_eight_frame_update_checks import train_updates
from train_symmetric_pilot_comparison import PilotDataset, make_model, write_json
from super_resolution_imaging.training.cli import seed_everything, sha256_file
from super_resolution_imaging.training.evaluation import confidence_interval


def main():
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1];dataset=root/'runs/four_layout_independent_128'
    selection_path=root/'data/four_layout_independent_128/selection.json'
    selection=json.loads(selection_path.read_text())
    if not (dataset/'COMPLETE.json').exists():raise ValueError('Generation incomplete')
    records=[json.loads(s) for s in (dataset/'manifest.jsonl').read_text().splitlines()]
    expected={r['sample_id']:r for r in selection['samples']}
    if len(records)!=128 or {r['sample_id'] for r in records}!=set(expected):raise ValueError('Scene set changed')
    brightness=[]
    for record in records:
        sid=record['sample_id'];folder=dataset/'samples'/sid;item=expected[sid]
        if record['split']!=item['split']:raise ValueError('Split changed')
        if sha256_file(folder/'source.png')!=item['source_png_sha256'] or sha256_file(folder/'target.png')!=item['target_sha256']:
            raise ValueError('Crop or GT changed')
        if record['output_sha256']['b'][:4]!=item['original_four_sha256']:raise ValueError('Initial frames changed')
        means=[]
        for group in ('a','b'):
            for i in range(8):
                path=folder/group/f'frame_{i:02d}.png'
                if sha256_file(path)!=record['output_sha256'][group][i]:raise ValueError('Frame hash mismatch')
                with Image.open(path) as im:
                    if im.mode!='L' or im.size!=(128,128):raise ValueError('Invalid frame dimensions')
                    if group=='b':means.append(float(np.asarray(im,dtype=np.float32).mean()/255.))
        brightness.append({'sample_id':sid,'max_new_center_mean_difference':max(abs(v-means[0]) for v in means[4:])})
    train=sorted(r['sample_id'] for r in records if r['split']=='train')
    val=sorted(r['sample_id'] for r in records if r['split']=='validation')
    if len(train)!=96 or len(val)!=32:raise ValueError('Unexpected split counts')
    def sources(ids):return {expected[sid]['source_id'] for sid in ids}
    old_sources=set(selection['excluded_previous_pilot_sources'])
    if sources(train)&sources(val) or (sources(train)|sources(val))&old_sources:raise ValueError('Source leakage')
    out=root/'runs/four_layout_independent_128_training';out.mkdir(exist_ok=True)
    write_json(out/'preflight.json',{'samples':128,'train':96,'validation':32,'previous_source_overlap':0,
        'source_hashes_verified':True,'max_new_center_mean_difference':max(r['max_new_center_mean_difference'] for r in brightness),
        'photometry_per_scene':brightness})
    if any(r['max_new_center_mean_difference']>.05 for r in brightness):raise ValueError('Photometric anomaly >0.05; inspect before training')
    protocol={'selection_sha256':sha256_file(selection_path),'manifest_sha256':sha256_file(dataset/'manifest.jsonl'),
        'dataset_config_sha256':sha256_file(dataset/'dataset.json'),'train_ids':train,'validation_ids':val,
        'seeds':[5400,5401,5402],'a_updates':1000,'b_updates':2000,'full_images':True,'batch_size':8,
        'variants':{'initial_four':[0,1,2,3],'quarter_four':[4,5,7,6]},
        'a_lr':3e-4,'b_lr':1e-4,'scheduler':'val L1 plateau every100 updates patience5 factor.5 min1e-6',
        'loss':'L1+.2*(1-SSIM)+.2*component','global_local_weights':0.,'weight_decay':1e-5,
        'primary':'fixed update2000 interior HR crop16','initialization':'independent A from scratch per seed; only 96 training sources',
        'no_formal_test':True,'limitation':'Two complete acquisition/reconstruction pipelines; PSF and bicubic base differ as well as positions'}
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Frozen protocol changed')
    write_json(out/'protocol.json',protocol)
    status={'status':'running','pid':os.getpid(),'completed_stages':[]};write_json(out/'status.json',status)
    try:
        for seed in protocol['seeds']:
            status.update(seed=seed,variant='a');write_json(out/'status.json',status)
            seed_everything(seed);model=make_model(4)
            center_train=PilotDataset(dataset,train,(0,1,2,3),center=True)
            center_val=PilotDataset(dataset,val,(0,1,2,3),center=True)
            shared=train_updates(model,center_train,center_val,out/f'seed{seed}_a',1000,3e-4,seed,8,
                {'protocol':protocol,'seed':seed,'variant':'a'})
            for name,indices in protocol['variants'].items():
                status.update(variant=name);write_json(out/'status.json',status)
                seed_everything(seed);model=make_model(4);model.load_state_dict(shared)
                train_data=PilotDataset(dataset,train,indices);val_data=PilotDataset(dataset,val,indices)
                train_updates(model,train_data,val_data,out/f'seed{seed}_{name}',2000,1e-4,seed,8,
                    {'protocol':protocol,'seed':seed,'variant':name})
                status['completed_stages'].append(f'{seed}_{name}');write_json(out/'status.json',status)
        results={kind:{name:{str(seed):json.loads((out/f'seed{seed}_{name}/{kind}_evaluation.json').read_text())
                           for seed in protocol['seeds']} for name in protocol['variants']} for kind in ('last','best')}
        means={kind:{name:{m:{k:float(np.mean([r['metrics'][m][k] for r in runs.values()]))
                    for k in ('psnr_db','ssim','l1','hf_rmse')} for m in ('0','16')}
                    for name,runs in variants.items()} for kind,variants in results.items()}
        paired={};per_seed=[]
        for seed in protocol['seeds']:
            per_seed.append(results['last']['quarter_four'][str(seed)]['metrics']['16']['psnr_db']-
                            results['last']['initial_four'][str(seed)]['metrics']['16']['psnr_db'])
        for metric in ('psnr_db','ssim','l1','hf_rmse'):
            delta=[]
            for sid in val:
                d=[]
                for seed in protocol['seeds']:
                    def value(name):return next(r[metric] for r in results['last'][name][str(seed)]['per_sample'] if r['margin']==16 and r['sample_id']==sid)
                    d.append(value('quarter_four')-value('initial_four'))
                delta.append(float(np.mean(d)))
            paired[metric]={'mean':float(np.mean(delta)),'ci95_mean':confidence_interval(np.array(delta)),
                'per_source_delta':dict(zip(val,delta)),'better_source_count':sum(v>0 if metric in ('psnr_db','ssim') else v<0 for v in delta)}
        summary={'results':results,'mean_metrics':means,'paired_quarter_minus_initial':paired,
            'per_seed_psnr_gain_db':per_seed,'scope':'Independent of earlier 32 pilot source images; exploratory validation, not formal test'}
        write_json(out/'summary.json',summary)
        lines=['# Independent 128-Scene Comparison of Two Four-Frame Layouts','', '96 training and 32 validation scenes across three seeds. The primary comparison uses 2,000 updates and removes 16 HR pixels per border.','',
            '|Layout|Full-image PSNR|Interior PSNR|Interior SSIM|Interior L1|Interior high-frequency RMSE|','|---|---:|---:|---:|---:|---:|']
        for name,m in means['last'].items():lines.append(f"|{name}|{m['0']['psnr_db']:.6f}|{m['16']['psnr_db']:.6f}|{m['16']['ssim']:.6f}|{m['16']['l1']:.6f}|{m['16']['hf_rmse']:.6f}|")
        lines.append(f'\nQuarter-pixel minus original four-frame PSNR by seed: {per_seed}.')
        lines.append('\nThe model runs and scenes are independent of the earlier pilot. Validation still guides scheduling, and the final test set was unused.')
        (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        status.update(status='completed');write_json(out/'status.json',status)
    except BaseException as exc:
        status.update(status='failed',error=repr(exc));write_json(out/'status.json',status);raise


if __name__=='__main__':main()
