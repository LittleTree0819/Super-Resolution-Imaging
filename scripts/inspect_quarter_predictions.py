import json,csv,hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw
from super_resolution_imaging.training.dataset import ZemaxBurstDataset
from super_resolution_imaging.training.evaluation import load_model,tensor_image
from super_resolution_imaging.training.model import gaussian_blur

root=Path(__file__).resolve().parents[1];out=root/'runs/quarter_four_prediction_inspection';out.mkdir(exist_ok=True)
torch.set_num_threads(4)
path=root/'runs/train_b_quarter_four_frozen_stage2/best.pt'
new,ck=load_model(path,'b',torch.device('cpu'));assert ck['epoch']==34
lock={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'epoch':34,'selection':'validation PSNR','test_used':False}
lockpath=out/'evaluation_checkpoint.json'
if lockpath.exists():assert json.loads(lockpath.read_text())==lock
else:lockpath.write_text(json.dumps(lock,indent=2))
old,_=load_model(root/'runs/train_b_expanded_split_residual_ssim_w020_stage2/best.pt','b',torch.device('cpu'))
datasets=[ZemaxBurstDataset(root/'runs'/d,'b','validation') for d in ('div2k_train_v1','div2k_quarter_four_v1')]
scores=list(csv.DictReader((root/'runs/quarter_four_validation_evaluation/per_sample.csv').open()))
deltas={r['sample_id']:float(r['quarter_best_psnr_db'])-float(r['original_best_psnr_db']) for r in scores}
ordered=sorted(deltas,key=deltas.get);bad=[s for s in ordered if deltas[s]<0]
# Best, median-positive and smallest-positive cases, chosen by explicit metric ranks.
positive=[s for s in ordered if deltas[s]>0];good=[positive[-1],positive[len(positive)//2],positive[0]]
rows=[];images={}
with torch.inference_mode():
 for i in range(72):
    a,b=[d[i] for d in datasets];sid=a['sample_id'];assert sid==b['sample_id'] and torch.equal(a['target'],b['target'])
    target=b['target'].unsqueeze(0);preds=[old(a['input'].unsqueeze(0)),new(b['input'].unsqueeze(0))]
    row={'sample_id':sid,'psnr_gain':deltas[sid]}
    for name,p in zip(('old','new'),preds):
        error=p-target;low=gaussian_blur(error,4);high=error-low
        row.update({name+'_signed_bias':error.mean().item(),name+'_abs_bias':error.mean().abs().item(),
                    name+'_mse':error.square().mean().item(),name+'_centered_mse':(error-error.mean()).square().mean().item(),
                    name+'_low_mse':low.square().mean().item(),name+'_high_mse':high.square().mean().item()})
    rows.append(row)
    if sid in bad+good:images[sid]=(target,preds)
def montage(ids,name):
 canvas=Image.new('L',(5*256,len(ids)*280),255);draw=ImageDraw.Draw(canvas)
 for j,sid in enumerate(ids):
    t,ps=images[sid];o,n=ps
    tiles=[t[0,0],o[0,0],n[0,0],(o-t).abs()[0,0]*4,(n-t).abs()[0,0]*4]
    for c,(label,tile) in enumerate(zip(('GT','old','new e34','old error x4','new error x4'),tiles)):
        canvas.paste(tensor_image(tile),(c*256,j*280+24));draw.text((c*256+3,j*280+4),f'{sid} {label} d={deltas[sid]:+.2f}',fill=0)
 canvas.save(out/name)
montage(bad[:5],'regressions_1.png');montage(bad[5:],'regressions_2.png');montage(good,'improvements.png')
with (out/'decomposition.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
report={'checkpoint':lock,'regressions':[r for r in rows if r['sample_id'] in bad],
 'typical_improvements':good,'definitions':'sigma=4 HR px; low=blur(pred-GT), high=error-low; energies not additive due cross term',
 'groups':{}}
for name,subset in [('all',rows),('regressions',[r for r in rows if r['sample_id'] in bad])]:
 report['groups'][name]={k:float(np.mean([r[k] for r in subset])) for k in rows[0] if k!='sample_id'}
(out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
