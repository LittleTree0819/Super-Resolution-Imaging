"""Convergence and worst-source brightness audit for independent layout replication."""
import json
from pathlib import Path
import numpy as np
import torch
from train_symmetric_pilot_comparison import PilotDataset, make_model, write_json


def main():
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1];out=root/'runs/four_layout_independent_128_training'
    summary=json.loads((out/'summary.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
    audit={'curves':{},'worst_source':'0637_03_00','worst_brightness':{}}
    for seed in protocol['seeds']:
        for variant in ('a','initial_four','quarter_four'):
            h=json.loads((out/f'seed{seed}_{variant}/history.json').read_text())
            best=max(h,key=lambda r:r['metrics']['16']['psnr_db'])
            audit['curves'][f'{seed}_{variant}']={'completed_updates':h[-1]['update'],
                'best_checked_update':best['update'],'best_interior_psnr':best['metrics']['16']['psnr_db'],
                'last_interior_psnr':h[-1]['metrics']['16']['psnr_db'],'last_lr':h[-1]['used_learning_rate']}
    for name,indices in protocol['variants'].items():
        data=PilotDataset(root/'runs/four_layout_independent_128',[audit['worst_source']],indices)
        b=data[0];bias=[]
        for seed in protocol['seeds']:
            state=torch.load(out/f'seed{seed}_{name}/last.pt',weights_only=False,map_location='cpu')
            model=make_model(4);model.load_state_dict(state['model_state']);model.eval()
            with torch.inference_mode():p=model(b['input'][None])[...,16:-16,16:-16]
            t=b['target'][None,...,16:-16,16:-16]
            e=p-t
            bias.append({'seed':seed,'mean_error':float(e.mean()),'centered_mse':float((e-e.mean()).square().mean())})
        audit['worst_brightness'][name]=bias
    audit['preflight']=json.loads((out/'preflight.json').read_text())
    audit['preflight'].pop('photometry_per_scene',None)
    write_json(out/'result_audit.json',audit)
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
