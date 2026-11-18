"""Geometric phase gate and nonlearning information diagnostics for eight frames."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from super_resolution_imaging.training.baselines import shift_and_add, metrics
from super_resolution_imaging.training.evaluation import confidence_interval
from super_resolution_imaging.data.writer import sha256_file
from analyze_nine_frame_shifts import fit_shift
from diagnose_nine_frame_pilot import translate, photo_match, highpass, canvas_save


def phase_geometry(path):
    geometry=json.loads((path/'geometry.json').read_text())
    fine=[r for r in geometry['rows'] if r['pupil_sampling']==33 and
          max(abs(v) for v in r['center_chief_xy_mm'])<=.1536]
    shifts=[];frames={}
    expected=((0,0),(.5,0),(0,-.5),(.5,-.5),(.25,.25),(-.25,.25),(.25,-.25),(-.25,-.25))
    for frame in range(8):
        selected=[r for r in fine if r['frame']==frame]
        if not selected or not all(r['chief_valid'] for r in selected):raise RuntimeError('Missing sensor-covered chief rays')
        values=np.array([r['chief_shift_png_lr_px'] for r in selected])
        shifts.append(tuple(values.mean(0)))
        frames[str(frame)]={'shift_mean_lr_px':values.mean(0).tolist(),'phase_mean':np.mod(values.mean(0),1).tolist(),
            'max_target_error_lr_px':float(np.linalg.norm(values-np.array(expected[frame]),axis=1).max())}
    a=np.array(shifts);delta=np.abs(np.mod(a[:,None,:]-a[None,:,:]+.5,1)-.5)
    distances=np.linalg.norm(delta,axis=2)
    new_min=float(distances[4:,:4].min())
    passed=new_min>.2 and max(frames[str(i)]['max_target_error_lr_px'] for i in range(4,8))<.03
    # Evaluate every matched field, not just averages.
    for r in [r for r in fine if r['frame']>=4]:
        old=[s for s in fine if s['frame']<4 and s['object_xy_mm']==r['object_xy_mm']]
        p=np.array(r['chief_shift_png_lr_px'])
        if min(np.linalg.norm(np.mod(p-np.array(s['chief_shift_png_lr_px'])+.5,1)-.5) for s in old)<=.2:passed=False
    result={'passed':passed,'frames':frames,'new_to_old_min_phase_distance_lr_px':new_min,
            'sensor_field_points':len(fine)//8,'limitations':'Chief-ray geometry establishes phases; does not prove recoverable image detail.'}
    (path/'phase_gate.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if not passed:raise RuntimeError('New phase geometry gate failed; generation blocked')
    return shifts,result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--geometry',type=Path,required=True)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();torch.set_num_threads(4)
    shifts,geometry=phase_geometry(args.geometry)
    print(json.dumps(geometry,indent=2),flush=True)
    if args.dataset is None:return
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    records=[json.loads(s) for s in (args.dataset/'manifest.jsonl').read_text().splitlines()]
    if len(records)!=32:raise RuntimeError('Expected 32 scenes')
    rows=[];structure=[];previews=[]
    root=Path(__file__).resolve().parents[1]
    for record in records:
        sid=record['sample_id'];sample=args.dataset/'samples'/sid
        for group in ('a','b'):
            for i,expected in enumerate(record['output_sha256'][group]):
                if sha256_file(sample/group/f'frame_{i:02d}.png')!=expected:raise RuntimeError('Frame hash mismatch')
        if sha256_file(sample/'target.png')!=record['target_sha256']:raise RuntimeError('Target hash mismatch')
        def load(path):return torch.tensor(np.asarray(Image.open(path),dtype=np.float32)/255.)
        frames=torch.stack([load(sample/'b'/f'frame_{i:02d}.png') for i in range(8)])
        target=load(sample/'target.png')[None,None]
        predictions={
            'four':shift_and_add(frames[:4][None],tuple(shifts[:4])),
            'new_four':shift_and_add(frames[4:][None],tuple(shifts[4:])),
            'eight':shift_and_add(frames[None],tuple(shifts)),
            'repeat_eight':shift_and_add(frames[:4].repeat(2,1,1)[None],tuple(shifts[:4])*2)}
        torch.testing.assert_close(predictions['four'],predictions['repeat_eight'],rtol=1e-5,atol=1e-6)
        row={'sample_id':sid}
        for name,pred in predictions.items():
            for region,border in [('full',0),('interior',16)]:
                p,t=(pred,target) if not border else (pred[...,border:-border,border:-border],target[...,border:-border,border:-border])
                for metric,value in zip(('l1','psnr_db','ssim'),metrics(p,t)):
                    row[f'{name}_{region}_{metric}']=float(value[0])
        baseline=root/'runs/nine_frame_pilot_32'/ 'samples'/sid
        row['target_matches_previous_pilot']=baseline.exists() and sha256_file(baseline/'target.png')==record['target_sha256']
        original=root/'runs/div2k_train_v1/samples'/sid.rsplit('_',1)[0]
        row['source_matches_original_four']=original.exists() and sha256_file(original/'source.png')==sha256_file(sample/'source.png')
        row['original_four_hashes_match']=original.exists() and all(sha256_file(original/'b'/f'frame_{i:02d}.png')==record['output_sha256']['b'][i] for i in range(4))
        rows.append(row)
        for i in range(4,8):
            distances=[np.linalg.norm(np.mod(np.array(shifts[i])-np.array(s)+.5,1)-.5) for s in shifts[:4]]
            near=int(np.argmin(distances))
            aligned=translate(frames[i],np.array(shifts[near])-np.array(shifts[i]))
            matched,gain,offset=photo_match(aligned,frames[near])
            difference=highpass(matched)-highpass(frames[near])
            mx,my,measured_gain,measured_offset,fit_mse=fit_shift(frames[0].numpy(),frames[i].numpy())
            structure.append({'sample_id':sid,'frame':i,'reference_frame':near,
                'mean_difference_from_center':float(frames[i].mean()-frames[0].mean()),
                'aligned_gain':gain,'aligned_offset':offset,'aligned_highpass_rmse':float(difference[8:-8,8:-8].square().mean().sqrt()),
                'image_fit_shift':[mx,my],'image_fit_gain':measured_gain,'image_fit_offset':measured_offset,
                'image_fit_mse':fit_mse,
                'note':'Nonzero aligned differences may reflect interpolation or PSF change, not necessarily useful new detail.'})
        if len(previews)<4:previews.append((sid,[frames[0],predictions['four'][0,0],predictions['eight'][0,0],target[0,0]]))
        print('Analyzed '+sid,flush=True)
    paired={}
    for region in ('full','interior'):
        values=np.array([r[f'eight_{region}_psnr_db']-r[f'four_{region}_psnr_db'] for r in rows])
        paired[region]={'mean_psnr_gain_db':float(values.mean()),'ci95_mean':confidence_interval(values),'improved':int((values>0).sum())}
    summary={'samples':32,'geometry':geometry,'fusion_paired_eight_minus_four':paired,
        'all_targets_match_previous_pilot':all(r['target_matches_previous_pilot'] for r in rows),
        'all_original_four_hashes_match':all(r['original_four_hashes_match'] for r in rows),
        'all_sources_match_original_four':all(r['source_matches_original_four'] for r in rows),
        'frame_mean_difference_max':max(abs(r['mean_difference_from_center']) for r in structure),
        'scope':'Geometric and nonlearning pilot only; no network training or held-out test claims.'}
    for filename,data in [('fusion_per_sample.csv',rows),('structure_per_frame.csv',structure)]:
        with (out/filename).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    canvas_save(out/'preview.png',previews,['B0','four fusion','eight fusion','target'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'report.md').write_text('# Eight-Frame Sampling-Phase Pilot\n\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n\nAdditional geometric sampling phases alone do not establish better reconstruction. Check interior fusion and per-image anomalies before running a network comparison.\n',encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
