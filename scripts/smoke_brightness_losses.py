"""One shared training batch, four loss configurations, no persistent model updates."""
import copy
import json
import tempfile
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from super_resolution_imaging.training.cli import run_epoch, save_checkpoint, sha256_file
from super_resolution_imaging.training.dataset import ZemaxBurstDataset, DEFAULT_EXCLUDED_IDS
from super_resolution_imaging.training.evaluation import load_model
from super_resolution_imaging.training.losses import ssim_loss, global_brightness_loss, local_low_frequency_loss
from super_resolution_imaging.training.model import frequency_split


def main():
    torch.set_num_threads(4)
    torch.manual_seed(5400)
    root = Path(__file__).resolve().parents[1]
    path = root / 'runs/train_b_expanded_split_residual_ssim_w020_stage2/best.pt'
    before_hash = sha256_file(path)
    initial, checkpoint = load_model(path, 'b', torch.device('cpu'))
    dataset = ZemaxBurstDataset(root / 'runs/div2k_train_v1', 'b', 'train', exclude_ids=DEFAULT_EXCLUDED_IDS)
    generator = torch.Generator().manual_seed(5400)
    batch = next(iter(DataLoader(dataset, batch_size=8, shuffle=True, generator=generator)))
    initial.train()
    pred, comp = initial.forward_with_components(batch['input'])
    ideal_low, ideal_high = frequency_split(batch['target']-comp['base'], 4)
    l1 = torch.nn.functional.l1_loss(pred, batch['target'])
    ssim = ssim_loss(pred, batch['target'])
    component = torch.nn.functional.l1_loss(comp['low'], ideal_low)+torch.nn.functional.l1_loss(comp['high'], ideal_high)
    base = l1 + 0.2*ssim + 0.2*component
    global_loss = global_brightness_loss(pred, batch['target'])
    local_loss = local_low_frequency_loss(pred, batch['target'], 4)
    terms = {'l1':l1,'ssim':ssim,'component':component,'base_objective':base,
             'global_brightness':global_loss,'local_low_frequency':local_loss}
    shared = list(initial.enc1.parameters())
    gradient_norms = {}
    for name, value in terms.items():
        assert torch.isfinite(value)
        grads = torch.autograd.grad(value, shared, retain_graph=True)
        assert all(torch.isfinite(g).all() for g in grads)
        gradient_norms[name] = float(torch.sqrt(sum(g.square().sum() for g in grads)))
    # An independent old-objective update must exactly match both new weights disabled.
    legacy = copy.deepcopy(initial)
    legacy_optimizer = torch.optim.AdamW(legacy.parameters(), lr=2.5e-5, weight_decay=1e-5)
    legacy_pred, legacy_comp = legacy.forward_with_components(batch['input'])
    lo, hi = frequency_split(batch['target']-legacy_comp['base'], 4)
    legacy_loss = (torch.nn.functional.l1_loss(legacy_pred,batch['target'])
                   +0.2*ssim_loss(legacy_pred,batch['target'])
                   +0.2*(torch.nn.functional.l1_loss(legacy_comp['low'],lo)
                         +torch.nn.functional.l1_loss(legacy_comp['high'],hi)))
    legacy_loss.backward(); legacy_optimizer.step()
    report = {'status':'passed', 'training_started':False, 'batch_samples':list(batch['sample_id']),
              'input_shape':list(batch['input'].shape),'target_shape':list(batch['target'].shape),
              'source_checkpoint_sha256':before_hash,
              'loss_values':{k:float(v.detach()) for k,v in terms.items()},
              'shared_encoder_gradient_norms':gradient_norms,'groups':{}}
    for name, gw, lw in [('C0',0,0),('C1',1,0),('C2',0,1),('C3',1,1)]:
        model = copy.deepcopy(initial)
        optimizer = torch.optim.AdamW(model.parameters(),lr=2.5e-5,weight_decay=1e-5)
        scores = run_epoch(model,[batch],torch.nn.L1Loss(),torch.device('cpu'),optimizer,
                           0.2,0.2,gw,lw,4)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        assert all(torch.isfinite(p).all() for p in model.parameters())
        change = max(float((p-initial.state_dict()[k]).abs().max()) for k,p in model.state_dict().items())
        assert change > 0
        if name=='C0':
            for p,q in zip(model.parameters(),legacy.parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=0)
        configuration = dict(checkpoint['configuration'],global_brightness_weight=gw,local_low_frequency_weight=lw)
        with tempfile.TemporaryDirectory(dir=root/'runs', prefix='brightness_smoke_tmp_') as temp:
            saved_path = Path(temp)/'smoke.pt'
            save_checkpoint(saved_path,{'model_state':model.state_dict(),'configuration':configuration})
            restored,saved = load_model(saved_path,'b',torch.device('cpu'))
            assert saved['configuration']['global_brightness_weight']==gw
            assert saved['configuration']['local_low_frequency_weight']==lw
            for p,q in zip(model.parameters(),restored.parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=0)
        report['groups'][name]={'weights':[gw,lw],'pre_update_l1':scores[0],
                              'pre_update_psnr_db':scores[1],'max_parameter_update':change,
                              'checkpoint_roundtrip':'passed'}
    raw = comp['base']+comp['low']+comp['high']
    report['pre_update_clip_fraction']=float(((raw<0)|(raw>1)).float().mean())
    report['zero_weights_match_legacy_update_exactly']=True
    assert sha256_file(path)==before_hash
    report['source_checkpoint_unchanged']=True
    out = root/'runs/brightness_loss_smoke_20261011'
    out.mkdir(exist_ok=False)
    (out/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
