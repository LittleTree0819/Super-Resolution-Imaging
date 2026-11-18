"""Check actual decay configuration, parameter equality, and float32 decay steps."""
import json
from pathlib import Path
import torch

torch.set_num_threads(4)
root=Path(__file__).resolve().parents[1]/'runs/weight_decay_comparison_20261014'
states={n:torch.load(root/n/'last.pt',weights_only=False,map_location='cpu') for n in ('R0','R1','R2')}
reference=states['R0']['model_state']
report={'checkpoints':{},'zero_gradient_AdamW_decay_probe':{}}
for name,state in states.items():
    report['checkpoints'][name]={
        'epoch':state['epoch'], 'configuration_decay':state['configuration']['weight_decay'],
        'optimizer_decay':state['optimizer_state']['param_groups'][0]['weight_decay'],
        'dtype':str(next(iter(state['model_state'].values())).dtype),
        'max_parameter_difference_vs_R0':max(float((v-reference[k]).abs().max()) for k,v in state['model_state'].items())}
for decay in (0.,1e-5,1e-4,1e-3):
    p=torch.nn.Parameter(torch.tensor([1.],dtype=torch.float32))
    optimizer=torch.optim.AdamW([p],lr=1e-4,weight_decay=decay)
    p.grad=torch.zeros_like(p);optimizer.step()
    report['zero_gradient_AdamW_decay_probe'][str(decay)]={
        'float32_factor':float(torch.tensor(1.-1e-4*decay,dtype=torch.float32)),
        'actual_parameter':float(p.detach())}
(root/'numerical_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
