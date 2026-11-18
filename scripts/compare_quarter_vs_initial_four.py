"""Direct paired analysis and channel-order equivalence for the two four-frame layouts."""
import json
from pathlib import Path
import numpy as np
import torch
from train_symmetric_pilot_comparison import PilotDataset, make_model, write_json
from super_resolution_imaging.training.evaluation import confidence_interval


def main():
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1];parent=root/'runs/eight_frame_four_controls'
    s=json.loads((parent/'summary.json').read_text())
    plan=json.loads((parent/'protocol.json').read_text())['parent_protocol']
    out=parent/'quarter_vs_initial';out.mkdir(exist_ok=True)
    report={'requested_positions_lr_px':[[.25,.25],[-.25,.25],[-.25,-.25],[.25,-.25]],
        'requested_dataset_indices':[4,5,7,6],'training_dataset_indices':[4,5,6,7],
        'original_positions_approx_lr_px':[[0,0],[.502499,0],[0,-.475736],[.502499,-.475736]],
        'scope':'Existing 24-train/8-validation, matched 2000 updates and three seeds; no new training or generation',
        'mean_metrics':{name:s['mean_metrics']['last'][name] for name in ('four','new_four')},'paired':{},'per_seed':[]}
    for seed in ('5400','5401','5402'):
        a=s['results']['last']['four'][seed]['metrics']['16']
        b=s['results']['last']['new_four'][seed]['metrics']['16']
        report['per_seed'].append({'seed':int(seed),'initial_psnr_db':a['psnr_db'],'quarter_psnr_db':b['psnr_db'],
            'psnr_delta_db':b['psnr_db']-a['psnr_db'],'ssim_delta':b['ssim']-a['ssim']})
    for metric in ('psnr_db','ssim','l1','hf_rmse'):
        values=[]
        for sid in plan['validation_ids']:
            delta=[]
            for seed in ('5400','5401','5402'):
                def value(name):return next(r[metric] for r in s['results']['last'][name][seed]['per_sample'] if r['sample_id']==sid and r['margin']==16)
                delta.append(value('new_four')-value('four'))
            values.append(float(np.mean(delta)))
        report['paired'][metric]={'mean':float(np.mean(values)),'ci95_mean':confidence_interval(np.array(values)),
            'better_source_count':sum(v>0 if metric in ('psnr_db','ssim') else v<0 for v in values),
            'per_source_delta':dict(zip(plan['validation_ids'],values))}
    # Permute both input channels and first-layer weights; B0 remains channel zero.
    data=PilotDataset(Path(plan['dataset']),plan['validation_ids'],(4,5,6,7))
    x=torch.stack([data[i]['input'] for i in range(len(data))])
    permutation=[0,1,3,2];errors={}
    with torch.inference_mode():
        for seed in ('5400','5401','5402'):
            checkpoint=torch.load(parent/f'seed{seed}_new_four/last.pt',map_location='cpu',weights_only=False)
            model=make_model(4);model.load_state_dict(checkpoint['model_state']);model.eval()
            original=model(x)
            reordered={k:v.clone() for k,v in checkpoint['model_state'].items()}
            reordered['enc1.0.weight']=reordered['enc1.0.weight'][:,permutation]
            model.load_state_dict(reordered)
            error=float((original-model(x[:,permutation])).abs().max())
            if error>1e-5:raise RuntimeError('Channel permutation equivalence failed')
            errors[seed]=error
    report['channel_order_equivalence_max_error']=errors
    write_json(out/'summary.json',report)
    lines=['# Quarter-Pixel vs. Original Four-Frame Sampling','', 'The two orders use the same set of positions. Reordering the final two frames preserves the output when the first-layer channels are permuted accordingly.','',
        '|Layout|Interior PSNR|Interior SSIM|Interior L1|Interior high-frequency RMSE|','|---|---:|---:|---:|---:|']
    for name in ('four','new_four'):
        m=report['mean_metrics'][name]['16']
        lines.append(f"|{name}|{m['psnr_db']:.6f}|{m['ssim']:.6f}|{m['l1']:.6f}|{m['hf_rmse']:.6f}|")
    lines.append('\nPaired statistics:\n'+json.dumps(report['paired'],ensure_ascii=False,indent=2))
    lines.append('\nThis pilot reuses eight validation source images. PSF, illumination and base-frame location also vary, so the comparison does not isolate sampling phase.')
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='mean_metrics'},indent=2))


if __name__=='__main__':main()
