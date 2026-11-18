"""Validation-only DC/low/high error decomposition and representative previews."""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from train_symmetric_pilot_comparison import PilotDataset, make_model, write_json
from super_resolution_imaging.training.model import gaussian_blur
from super_resolution_imaging.training.baselines import metrics
from super_resolution_imaging.training.cli import sha256_file
from diagnose_nine_frame_pilot import translate, photo_match
from super_resolution_imaging.training.evaluation import tensor_image


def main():
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1];parent=root/'runs/four_layout_independent_128_training'
    protocol=json.loads((parent/'protocol.json').read_text());comparison=json.loads((parent/'summary.json').read_text())
    out=parent/'brightness_low_frequency_diagnosis';out.mkdir(exist_ok=True)
    data=root/'runs/four_layout_independent_128'
    ids=protocol['validation_ids']
    shifts=json.loads((root/'runs/eight_frame_quarter_geometry/phase_gate.json').read_text())['frames']
    datasets={n:PilotDataset(data,ids,indices) for n,indices in protocol['variants'].items()}
    rows=[];photos=[];examples={};hashes={}
    deltas=comparison['paired_quarter_minus_initial']['psnr_db']['per_source_delta']
    ordered=sorted(ids,key=lambda sid:deltas[sid]);selected=ordered[:4]+ordered[-2:][::-1]
    def crop(x):return x[...,16:-16,16:-16]
    def avg(x):return x.flatten(1).mean(1)
    def load(path):return torch.from_numpy(np.asarray(Image.open(path),dtype=np.float32)/255.)
    for sid in ids:
        folder=data/'samples'/sid
        center=load(folder/'b/frame_00.png')
        for frame in range(8):
            current=load(folder/'b'/f'frame_{frame:02d}.png')
            xy=np.array(shifts[str(frame)]['shift_mean_lr_px'])
            aligned=translate(current,-xy)
            matched,gain,offset=photo_match(aligned,center)
            photos.append({'sample_id':sid,'frame':frame,'input_mean':float(current.mean()),
                'raw_mean_delta_vs_center':float(current.mean()-center.mean()),
                'aligned_gain_to_center':gain,'aligned_offset_to_center':offset,
                'aligned_matched_rmse':float((matched[8:-8,8:-8]-center[8:-8,8:-8]).square().mean().sqrt())})
    for kind in ('last','best'):
        for seed in protocol['seeds']:
            for name,ds in datasets.items():
                path=parent/f'seed{seed}_{name}/{kind}.pt';hashes[f'{seed}_{name}_{kind}']=sha256_file(path)
                state=torch.load(path,weights_only=False,map_location='cpu')
                model=make_model(4);model.load_state_dict(state['model_state']);model.eval()
                with torch.inference_mode():
                    for start in range(0,len(ds),4):
                        items=[ds[i] for i in range(start,min(start+4,len(ds)))]
                        x=torch.stack([b['input'] for b in items]);target=torch.stack([b['target'] for b in items])
                        pred,comp=model.forward_with_components(x);e=pred-target
                        error=crop(e);low=crop(gaussian_blur(e,4));high=error-low;bias=avg(error)
                        truth_low=gaussian_blur(target-comp['base'],4)
                        raw=comp['base']+comp['low']+comp['high']
                        l1,psnr,ssim=metrics(crop(pred),crop(target))
                        oracle=(crop(pred)-bias[:,None,None,None]).clamp(0,1)
                        quantities={'l1':l1,'psnr_db':psnr,'ssim':ssim,'bias':bias,'abs_bias':bias.abs(),
                            'mse':avg(error.square()),'dc_mse':bias.square(),
                            'centered_mse':avg((error-bias[:,None,None,None]).square()),
                            'low_mse':avg(low.square()),'local_low_mse':avg((low-low.mean((1,2,3),keepdim=True)).square()),
                            'high_mse':avg(high.square()),'low_high_cross':2*avg(low*high),
                            'low_head_mae':avg(crop(comp['low']-truth_low).abs()),
                            'base_mean':avg(crop(comp['base'])),'low_head_mean':avg(crop(comp['low'])),
                            'desired_low_mean':avg(crop(truth_low)),'high_head_mean':avg(crop(comp['high'])),
                            'clip_bias':avg(crop(pred-raw)),'clip_fraction':avg(crop(((raw<0)|(raw>1)).float())),
                            'oracle_dc_psnr_db':metrics(oracle,crop(target))[1],
                            'target_mean':avg(crop(target)),'target_std':crop(target).flatten(1).std(1)}
                        torch.testing.assert_close(quantities['mse'],quantities['dc_mse']+quantities['centered_mse'])
                        torch.testing.assert_close(quantities['mse'],quantities['low_mse']+quantities['high_mse']+quantities['low_high_cross'])
                        for i,b in enumerate(items):
                            sid=b['sample_id'];rows.append({'sample_id':sid,'seed':seed,'checkpoint':kind,'variant':name,
                                **{k:float(v[i]) for k,v in quantities.items()}})
                            if seed==5401 and sid in selected:
                                ex=examples.setdefault((kind,sid),{'target':target[i,0].clone()})
                                ex[name]=pred[i,0].clone();ex[name+'_low_error']=gaussian_blur(e[i:i+1],4)[0,0]
                                ex[name+'_high_error']=e[i,0]-ex[name+'_low_error']
                print(f'Analyzed {kind} seed={seed} {name}',flush=True)
    summary={'samples':32,'seeds':protocol['seeds'],'sigma_hr_px':4,'crop_hr':16,'checkpoint_sha256':hashes,
             'mean_metrics':{},'degraded_sources':[sid for sid in ids if deltas[sid]<0], 'degraded_details':{},
             'selected_preview_ids':selected,'preview_seed':5401,'subgroups':{}}
    keys=[k for k in rows[0] if k not in ('sample_id','seed','checkpoint','variant')]
    def means(subset):return {k:float(np.mean([r[k] for r in subset])) for k in keys}
    for kind in ('last','best'):
        summary['mean_metrics'][kind]={name:means([r for r in rows if r['checkpoint']==kind and r['variant']==name]) for name in datasets}
    for label,selected_ids in [('degraded',summary['degraded_sources']),('improved',[sid for sid in ids if deltas[sid]>0])]:
        summary['subgroups'][label]={name:means([r for r in rows if r['checkpoint']=='last' and r['variant']==name and r['sample_id'] in selected_ids]) for name in datasets}
    for sid in summary['degraded_sources']:
        summary['degraded_details'][sid]={kind:{name:means([r for r in rows if r['sample_id']==sid and r['variant']==name and r['checkpoint']==kind]) for name in datasets} for kind in ('last','best')}
    summary['photometry']={'max_raw_new_center_mean_difference':max(abs(r['raw_mean_delta_vs_center']) for r in photos if r['frame']>=4),
        'raw_max_case':max(photos,key=lambda r:abs(r['raw_mean_delta_vs_center'])),
        'new_gain_range':[min(r['aligned_gain_to_center'] for r in photos if r['frame']>=4),max(r['aligned_gain_to_center'] for r in photos if r['frame']>=4)],
        'note':'Mean changes include shifted content and PSF/illumination; not a pure exposure-drift measurement'}
    for filename,data_rows in [('per_scene_seed.csv',rows),('input_photometry.csv',photos)]:
        with (out/filename).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(data_rows[0]));writer.writeheader();writer.writerows(data_rows)
    for kind in ('last','best'):
        labels=('target','initial_four','quarter_four','initial_four_low_error','quarter_four_low_error',
                'initial_four_high_error','quarter_four_high_error')
        canvas=Image.new('RGB',(7*256,280*len(selected)),'white');draw=ImageDraw.Draw(canvas)
        for row,sid in enumerate(selected):
            for col,name in enumerate(labels):
                v=examples[(kind,sid)][name]
                if col<3:tile=tensor_image(v).convert('RGB')
                elif col<5:
                    a=np.clip(v.numpy()/.1,-1,1)
                    rgb=np.stack((1-np.maximum(-a,0),1-np.abs(a),1-np.maximum(a,0)),axis=-1)
                    tile=Image.fromarray(np.uint8(rgb*255))
                else:tile=tensor_image(v.abs()*8).convert('RGB')
                canvas.paste(tile,(col*256,row*280+24));draw.text((col*256+3,row*280+4),sid+' '+name,fill='black')
        canvas.save(out/f'{kind}_preview.png')
    write_json(out/'summary.json',summary)
    print(json.dumps({'mean_metrics':summary['mean_metrics'],'photometry':summary['photometry']},indent=2))


if __name__=='__main__':main()
