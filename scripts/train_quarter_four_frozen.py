"""Formal B-stage launcher bound to the smoke-tested protocol."""
import json
import sys
from pathlib import Path
import torch
from super_resolution_imaging.training.cli import main, sha256_file

root=Path(__file__).resolve().parents[1]
cfg=json.loads((root/'configs/quarter_four_training_frozen.json').read_text())
assert sha256_file(Path(cfg['dataset'])/'manifest.jsonl')==cfg['manifest_sha256']
assert sha256_file(Path(cfg['initialize_from']))==cfg['checkpoint_sha256']
assert json.loads((root/'runs/quarter_four_training_smoke/report.json').read_text())['status']=='passed'
torch.set_num_threads(4)
output=root/'runs/train_b_quarter_four_frozen_stage2'
args=['--dataset',cfg['dataset'],'--output',str(output),'--group','b',
      '--split-residual-output','--epochs','100','--device','cpu',
      '--early-stopping-patience','20']
for key in ('base_channels','split_sigma','learning_rate','weight_decay','batch_size',
            'component_loss_weight','ssim_loss_weight','global_brightness_weight',
            'local_low_frequency_weight','seed'):
    args += ['--'+key.replace('_','-'),str(cfg[key])]
args += ['--exclude']+cfg['exclude_ids']
if (output/'last.pt').exists():args += ['--resume',str(output/'last.pt')]
else:args += ['--initialize-from',cfg['initialize_from']]
print('Frozen B-stage: maximum 100 epochs, patience 20, no test evaluation; '+str(output),flush=True)
sys.argv=[sys.argv[0]]+args
main()
