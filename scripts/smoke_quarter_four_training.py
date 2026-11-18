"""Frozen quarter-four preflight; two disposable updates, never formal training."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from super_resolution_imaging.training.dataset import ZemaxBurstDataset, DEFAULT_EXCLUDED_IDS
from super_resolution_imaging.training.evaluation import load_model
from super_resolution_imaging.training.cli import run_epoch, seed_everything, sha256_file


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'runs/quarter_four_training_smoke';out.mkdir(exist_ok=True)
    protocol=root/'configs/quarter_four_training_frozen.json'
    dataset=root/'runs/div2k_quarter_four_v1'
    source=root/'runs/train_a_expanded_split_residual_ssim_w020_stage1/best.pt'
    excluded=list(DEFAULT_EXCLUDED_IDS)+['0276_03','0755_00']
    plan=json.loads((root/'configs/quarter_four_release_plan.json').read_text())
    cfg={'dataset':str(dataset),'manifest_sha256':sha256_file(dataset/'manifest.jsonl'),
         'initialize_from':str(source),'checkpoint_sha256':sha256_file(source),
         'exclude_ids':excluded,'counts':{'train':2584,'validation':72,'test':80},
         'channel_shifts_lr_px':plan['target_image_shifts_lr_px'],
         'base_channels':24,'split_residual_output':True,'split_sigma':4.0,
         'learning_rate':1e-4,'weight_decay':1e-5,'batch_size':8,'patch_size':None,
         'component_loss_weight':.2,'ssim_loss_weight':.2,
         'global_brightness_weight':0.,'local_low_frequency_weight':0.,
         'scheduler':{'kind':'ReduceLROnPlateau','monitor':'validation_l1','patience':5,'factor':.5,'min_lr':1e-6},
         'seed':5400,'formal_training_started':False}
    assert cfg['checkpoint_sha256']=='86afbf8a198b8c9565c0aff39b856751301d30242baf8f13611e262f9684a86e'
    if protocol.exists():assert json.loads(protocol.read_text())==cfg,'Frozen protocol changed'
    else:protocol.write_text(json.dumps(cfg,indent=2)+'\n')
    torch.set_num_threads(4);seed_everything(5400)
    datasets={s:ZemaxBurstDataset(dataset,'b',s,exclude_ids=tuple(excluded)) for s in cfg['counts']}
    for split,data in datasets.items():
        assert len(data)==cfg['counts'][split]
        assert not set(excluded)&{r['sample_id'] for r in data.records}
        assert all(r['positions_b_mm']==plan['lens_positions_mm'] for r in data.records)
    # Decode training/validation only; test is limited to manifest counts.
    for data in (datasets['train'],datasets['validation']):
        item=data[0];folder=dataset/'samples'/item['sample_id']
        assert item['input'].shape==(4,128,128) and item['target'].shape==(1,256,256)
        for i in range(4):
            with Image.open(folder/'b'/f'frame_{i:02d}.png') as im:
                assert torch.equal(item['input'][i],torch.from_numpy(np.array(im,dtype=np.float32)/255.))
    model,saved=load_model(source,'a',torch.device('cpu'))
    assert saved['configuration']['base_channels']==24 and model.split_residual_output and model.split_sigma==4.
    assert all(torch.equal(v,saved['model_state'][k]) for k,v in model.state_dict().items())
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,factor=.5,min_lr=1e-6)
    train=DataLoader(Subset(datasets['train'],range(8)),batch_size=8)
    validation=DataLoader(Subset(datasets['validation'],range(2)),batch_size=2)
    before={k:v.clone() for k,v in model.state_dict().items()}
    results=[]
    for step in range(2):
        metrics=run_epoch(model,train,torch.nn.L1Loss(),torch.device('cpu'),optimizer,
                          component_loss_weight=.2,ssim_loss_weight=.2,
                          global_brightness_weight=0.,local_low_frequency_weight=0.)
        assert np.isfinite(metrics).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        results.append(metrics);print(f'Smoke update {step+1}/2: {metrics}',flush=True)
    assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items())
    val=run_epoch(model,validation,torch.nn.L1Loss(),torch.device('cpu'));assert np.isfinite(val).all()
    scheduler.step(val[0])
    path=out/'disposable_checkpoint.pt'
    torch.save({'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),'scheduler_state':scheduler.state_dict()},path)
    restored=torch.load(path,weights_only=False)
    assert all(torch.equal(v,restored['model_state'][k]) for k,v in model.state_dict().items())
    assert sha256_file(source)==cfg['checkpoint_sha256']
    report={'status':'passed','formal_training_started':False,'protocol':str(protocol),
            'counts':cfg['counts'],'channel_order':'frame_00..03 exactly matches frozen positions',
            'checkpoint_load':'strict exact state match','checkpoint_roundtrip':'passed',
            'loss':'L1 + 0.2*(1-SSIM) + 0.2*component; global/local disabled',
            'smoke_updates':2,'train_metrics':results,'validation_metrics':val,'test_images_loaded':False}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
