"""Read-only validation diagnosis of brightness and split residual errors."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from super_resolution_imaging.training.dataset import ZemaxBurstDataset, DEFAULT_EXCLUDED_IDS
from super_resolution_imaging.training.evaluation import load_model, tensor_image
from super_resolution_imaging.training.baselines import metrics
from super_resolution_imaging.training.model import gaussian_blur


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parents[1]
    out = root / 'runs/validation_brightness_analysis_20261011'
    out.mkdir(exist_ok=False)
    paths = {name: root / f'runs/{folder}/best.pt' for name, folder in {
        'old_b': 'train_b_split_residual_ssim_w020_stage2',
        'new_b': 'train_b_expanded_split_residual_ssim_w020_stage2'}.items()}
    models = {k: load_model(p, 'b', torch.device('cpu'))[0] for k, p in paths.items()}
    ds = ZemaxBurstDataset(root / 'runs/div2k_train_v1', 'b', 'validation', exclude_ids=DEFAULT_EXCLUDED_IDS)
    rows, examples = [], []
    def mean(x): return x.flatten(1).mean(1)
    with torch.inference_mode():
        for batch in DataLoader(ds, batch_size=8, shuffle=False):
            target = batch['target']
            row_batch = [{'sample_id': sid, 'target_mean': float(mean(target)[i])}
                         for i, sid in enumerate(batch['sample_id'])]
            preds, components = {}, {}
            for name, model in models.items():
                pred, comp = model.forward_with_components(batch['input'])
                preds[name], components[name] = pred, comp
                error = pred - target
                bias = mean(error)
                low_error = gaussian_blur(error, model.split_sigma)
                high_error = error - low_error
                residual = target - comp['base']
                desired_low = gaussian_blur(residual, model.split_sigma)
                desired_high = residual - desired_low
                raw = comp['base'] + comp['low'] + comp['high']
                corrected = (pred - bias[:, None, None, None]).clamp(0, 1)
                score, corrected_score = metrics(pred, target), metrics(corrected, target)
                quantities = {
                    'bias': bias, 'abs_bias': bias.abs(), 'mse': mean(error.square()),
                    'dc_mse': bias.square(), 'centered_mse': mean((error-bias[:,None,None,None]).square()),
                    'low_mse': mean(low_error.square()), 'high_mse': mean(high_error.square()),
                    'low_high_cross': 2*mean(low_error*high_error),
                    'low_centered_mse': mean((low_error-bias[:,None,None,None]).square()),
                    'low_head_mae': mean((comp['low']-desired_low).abs()),
                    'high_head_mae': mean((comp['high']-desired_high).abs()),
                    'base_mean': mean(comp['base']), 'low_mean': mean(comp['low']),
                    'desired_low_mean': mean(desired_low), 'high_mean': mean(comp['high']),
                    'clip_bias': mean(pred-raw), 'clip_fraction': mean(((raw<0)|(raw>1)).float()),
                    'oracle_dc_psnr_gain': corrected_score[1]-score[1],
                }
                for metric, values in zip(('l1', 'psnr_db', 'ssim'), score): quantities[metric] = values
                for i, row in enumerate(row_batch):
                    for key, values in quantities.items(): row[f'{name}_{key}'] = float(values[i])
                # Check both decompositions, retaining the low/high cross term.
                torch.testing.assert_close(mean(error.square()), bias.square()+quantities['centered_mse'])
                torch.testing.assert_close(mean(error.square()), quantities['low_mse']+quantities['high_mse']+quantities['low_high_cross'])
            for i, row in enumerate(row_batch):
                row['delta_psnr_db'] = row['new_b_psnr_db']-row['old_b_psnr_db']
                rows.append(row)
                examples.append({'target': target[i,0].clone(),
                    'old': preds['old_b'][i,0].clone(), 'new': preds['new_b'][i,0].clone(),
                    'low_error': gaussian_blur(preds['new_b'][i:i+1]-target[i:i+1], 4)[0,0],
                    'low': components['new_b']['low'][i,0].clone(),
                    'desired_low': gaussian_blur(target[i:i+1]-components['new_b']['base'][i:i+1], 4)[0,0]})
    with (out/'per_sample.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={'split':'validation', 'samples':len(rows), 'sigma_hr_pixels':4,
             'checkpoint_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
             'metrics':{}, 'subgroups':{}}
    def summarize(indices):
        return {name: {key:float(np.mean([rows[i][f'{name}_{key}'] for i in indices]))
                for key in ('l1','psnr_db','ssim','abs_bias','mse','dc_mse','centered_mse',
                            'low_mse','high_mse','low_high_cross','low_centered_mse',
                            'low_head_mae','high_head_mae','oracle_dc_psnr_gain','clip_fraction','clip_bias')}
                for name in models}
    summary['metrics']=summarize(range(len(rows)))
    for label, indices in [('improved',[i for i,r in enumerate(rows) if r['delta_psnr_db']>0]),
                           ('degraded',[i for i,r in enumerate(rows) if r['delta_psnr_db']<0])]:
        summary['subgroups'][label]={'samples':len(indices), 'metrics':summarize(indices)}
    degraded=[r for r in rows if r['delta_psnr_db']<0]
    summary['degraded_error_increases']={key:sum(r[f'new_b_{key}']>r[f'old_b_{key}'] for r in degraded)
                                         for key in ('dc_mse','low_mse','high_mse','low_head_mae','high_head_mae')}
    summary['correlation_delta_mse']={key:float(np.corrcoef(
        [r['new_b_mse']-r['old_b_mse'] for r in rows],
        [r[f'new_b_{key}']-r[f'old_b_{key}'] for r in rows])[0,1])
        for key in ('dc_mse','low_mse','high_mse')}
    order=sorted(range(len(rows)), key=lambda i:rows[i]['delta_psnr_db'])
    summary['worst_samples']=[rows[i] for i in order[:5]]
    selected=order[:3]+order[-2:][::-1]
    labels=('old','new','target','low','desired_low','low_error')
    canvas=Image.new('RGB',(256*6,280*len(selected)), 'white'); draw=ImageDraw.Draw(canvas)
    for row,index in enumerate(selected):
        ex=examples[index]
        for col,key in enumerate(labels):
            if col<3: tile=tensor_image(ex[key]).convert('RGB')
            else:
                # Fixed signed scale: blue negative, red positive, white zero; +/-0.10 saturates.
                v=np.clip(ex[key].numpy()/0.10,-1,1)
                rgb=np.stack((1-np.maximum(-v,0),1-np.abs(v),1-np.maximum(v,0)),axis=-1)
                tile=Image.fromarray(np.uint8(rgb*255))
            canvas.paste(tile,(col*256,row*280+24))
            draw.text((col*256+3,row*280+4),f"{rows[index]['sample_id']} {key}",fill='black')
    canvas.save(out/'diagnostic_preview.png')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='worst_samples'},indent=2),flush=True)


if __name__=='__main__': main()
