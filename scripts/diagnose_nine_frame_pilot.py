"""Registration diagnostics, paired fusion controls and aligned frame differences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from analyze_nine_frame_shifts import TARGET_SHIFTS, fit_shift
from super_resolution_imaging.training.baselines import bicubic, metrics, shift_and_add

FOUR = (0, 1, 7, 8)
# Same phase multiplicities as nine positions, using only the original frames.
REPEATED = (0, 1, 8, 7, 8, 1, 8, 7, 8)
PAIRS = {2: 8, 3: 7, 4: 8, 5: 1, 6: 8}


def load(path):
    return torch.from_numpy(np.asarray(Image.open(path).convert('L'), dtype=np.float32) / 255.)


def translate(image, xy):
    h, w = image.shape
    y, x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
    grid = torch.stack((x - 2 * xy[0] / (w - 1), y - 2 * xy[1] / (h - 1)), -1)[None]
    return F.grid_sample(image[None, None], grid, mode='bicubic', padding_mode='reflection', align_corners=True)[0, 0]


def photo_match(image, reference):
    a, b = image[8:-8, 8:-8], reference[8:-8, 8:-8]
    gain = ((a-a.mean())*(b-b.mean())).mean() / (a-a.mean()).square().mean().clamp_min(1e-12)
    offset = b.mean() - gain*a.mean()
    return image*gain+offset, float(gain), float(offset)


def highpass(image):
    low = F.avg_pool2d(image[None, None], 5, stride=1, padding=2)[0, 0]
    return image-low


def stats(values):
    a = np.asarray(values)
    return {'mean': float(a.mean()), 'median': float(np.median(a)), 'min': float(a.min()), 'max': float(a.max())}


def canvas_save(path, rows, labels, signed=False):
    canvas = Image.new('RGB', (len(labels)*256, len(rows)*280), 'white')
    draw = ImageDraw.Draw(canvas)
    for row, (sid, images) in enumerate(rows):
        for col, im in enumerate(images):
            value = (im/.02*.5+.5) if signed and col >= 2 else im
            tile = Image.fromarray(np.uint8(value.clamp(0,1).numpy()*255)).resize((256,256))
            canvas.paste(tile,(col*256,row*280+24))
            draw.text((col*256+4,row*280+4),sid+' '+labels[col],fill='black')
    canvas.save(path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',type=Path,default=Path('runs/nine_frame_pilot_32'))
    p.add_argument('--output',type=Path,default=Path('runs/nine_frame_pilot_diagnostics'))
    args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    report=json.loads((args.dataset/'shift_report.json').read_text())
    calibration=json.loads(Path('runs/nine_frame_calibration_refined/shift_report.json').read_text())
    fixed=tuple(tuple(calibration['frames'][str(i)]['median_measured_shift_xy']) for i in range(9))
    population=tuple(tuple(report['frames'][str(i)]['median_measured_shift_xy']) for i in range(9))
    records=[json.loads(s) for s in (args.dataset/'manifest.jsonl').read_text().splitlines()]
    measured={sid:tuple(tuple(r['measured_shift_xy']) for r in sorted(report['rows'],key=lambda r:r['frame']) if r['sample_id']==sid) for sid in [r['sample_id'] for r in records]}
    anomaly_ids=('0677_03_00','0193_02_00','0334_02_00','0100_00_00')
    fusion=[]; structure=[]; anomalies=[]; previews=[]; differences=[]
    for record in records:
        sid=record['sample_id']; sample=args.dataset/'samples'/sid
        frames=torch.stack([load(sample/'b'/f'frame_{i:02d}.png') for i in range(9)])
        target=load(sample/'target.png')[None,None]
        per_scene=measured[sid]
        # Conservative fallback: large disagreement with scene-independent calibration.
        reliable=tuple(per_scene[i] if np.linalg.norm(np.subtract(per_scene[i],fixed[i]))<=.15 else fixed[i] for i in range(9))
        predictions={'bicubic':bicubic(frames[None,:1]),'mean4':bicubic(frames[list(FOUR)].mean(0)[None,None]),'mean9':bicubic(frames.mean(0)[None,None])}
        for name, shifts in [('ideal',TARGET_SHIFTS),('fixed',fixed),('population',population),('scene',per_scene),('guarded',reliable)]:
            for n,inds in [(4,FOUR),(9,tuple(range(9)))]:
                predictions[f'{name}_{n}']=shift_and_add(frames[None,list(inds)],tuple(shifts[i] for i in inds))
        predictions['repeated4_as9']=shift_and_add(frames[None,list(REPEATED)],tuple(fixed[i] for i in REPEATED))
        for name,pred in predictions.items():
            for margin in (0,16):
                a,b=(pred,target) if margin==0 else (pred[:,:,margin:-margin,margin:-margin],target[:,:,margin:-margin,margin:-margin])
                l1,psnr,ssim=metrics(a,b)
                fusion.append({'sample_id':sid,'method':name,'hr_crop_margin':margin,'l1':float(l1[0]),'psnr':float(psnr[0]),'ssim':float(ssim[0])})
        aligned=[translate(frames[i],(-fixed[i][0],-fixed[i][1])) for i in range(9)]
        matched=[photo_match(f,aligned[0])[0] for f in aligned]
        for i,j in PAIRS.items():
            raw=(aligned[i]-aligned[j])[8:-8,8:-8]
            diff=(matched[i]-matched[j])[8:-8,8:-8]
            hp=highpass(matched[i]-matched[j])[8:-8,8:-8]
            basehp=highpass(matched[j])[8:-8,8:-8]
            structure.append({'sample_id':sid,'new_frame':i,'same_phase_original_frame':j,'raw_rms':float(raw.square().mean().sqrt()),'photo_matched_rms':float(diff.square().mean().sqrt()),'highpass_rms':float(hp.square().mean().sqrt()),'reference_highpass_rms':float(basehp.square().mean().sqrt())})
        if sid in anomaly_ids:
            for i in (1,3,5,7):
                local=[]
                for y,x in ((12,12),(12,68),(68,12),(68,68)):
                    ref=frames[0,y:y+48,x:x+48].numpy()*255
                    mov=frames[i,y:y+48,x:x+48].numpy()*255
                    result=fit_shift(ref,mov)
                    local.append({'region_xy':[x,y],'reference_std_8bit':float(ref.std()),'shift_xy':result[:2],'gain':result[2],'offset_8bit':result[3],'mse_8bit':result[4]})
                anomalies.append({'sample_id':sid,'frame':i,'global_shift':per_scene[i],'fixed_shift':fixed[i],'local_fits':local})
            previews.append((sid,[predictions['bicubic'][0,0],predictions['fixed_4'][0,0],predictions['fixed_9'][0,0],predictions['scene_9'][0,0],target[0,0]]))
            differences.append((sid,[frames[0],matched[5],matched[5]-matched[1],matched[3]-matched[7],matched[2]-matched[8]]))
        print('completed diagnostics: '+sid,flush=True)
    methods=list(dict.fromkeys(r['method'] for r in fusion))
    summary={'samples':len(records),'fusion':{},'paired_gains':{},'structure':{},'notes':['Scene shifts fit B against B0 without GT; guarded uses fixed calibration if disagreement exceeds 0.15 LR px.','Population shifts are descriptive estimates on this pilot, not independent calibration.','Difference images use fixed calibration and affine brightness matching; they include interpolation/registration error and do not by themselves establish recoverable information.','HR crop 16 removes 8 LR pixels on each edge; full-image metrics also retained.','Repeated4_as9 duplicates original frames in 1:2:2:4 phase counts to isolate phase-weight effects.']}
    for margin in (0,16):
        selected=[r for r in fusion if r['hr_crop_margin']==margin]
        summary['fusion'][str(margin)]={m:{k:float(np.mean([r[k] for r in selected if r['method']==m])) for k in ('l1','psnr','ssim')} for m in methods}
        lookup={(r['sample_id'],r['method']):r for r in selected}
        for name,a,b in [('fixed9_minus_fixed4','fixed_9','fixed_4'),('fixed9_minus_bicubic','fixed_9','bicubic'),('fixed9_minus_repeated4','fixed_9','repeated4_as9'),('scene9_minus_scene4','scene_9','scene_4'),('guarded9_minus_guarded4','guarded_9','guarded_4')]:
            d=[lookup[(r['sample_id'],a)]['psnr']-lookup[(r['sample_id'],b)]['psnr'] for r in records]
            summary['paired_gains'][f'{name}_crop{margin}']={**stats(d),'positive_count':sum(v>0 for v in d)}
    for i in PAIRS:
        rs=[r for r in structure if r['new_frame']==i]
        summary['structure'][str(i)]={k:stats([r[k] for r in rs]) for k in ('raw_rms','photo_matched_rms','highpass_rms','reference_highpass_rms')}
    for name,data in [('summary',summary),('fusion_per_sample',fusion),('structure_per_sample',structure),('anomaly_local_registration',anomalies)]:
        (out/f'{name}.json').write_text(json.dumps(data,indent=2)+'\n')
    canvas_save(out/'anomaly_fusion.png',previews,['Bicubic','fixed4','fixed9','scene9','GT'])
    canvas_save(out/'aligned_differences.png',differences,['B0','aligned W','W-E diff','N-S diff','NE-SE diff'],True)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    with torch.inference_mode():
        main()
