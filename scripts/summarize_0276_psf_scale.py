"""Summarize isolated controls and draw a shared-scale diagnostic preview."""
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

root=Path(__file__).resolve().parents[1]
out=root/'runs/0276_03_psf_scale_check'
r=json.loads((out/'report.json').read_text(encoding='utf-8'))
f=json.loads((out/'followup/report.json').read_text(encoding='utf-8'))
sample=root/'runs/div2k_quarter_four_v1/samples/0276_03'
metadata=json.loads((sample/'metadata.json').read_text(encoding='utf-8'))
verified={}
for group in ('a','b'):
    for i,expected in enumerate(metadata['output_sha256'][group]):
        path=sample/group/f'frame_{i:02d}.png'
        verified[f'{group}/{path.name}']=hashlib.sha256(path.read_bytes()).hexdigest()==expected
def read(path): return np.asarray(Image.open(path).convert('L'),dtype=float)/255
summary={'completed':bool(r['completed'] and f['completed']),
         'original_frame_hashes_match_metadata':verified,
         'dataset_modified':False,'production_generation_resumed':False,
         'source':'https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v251/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Image_Simulation.html',
         'normalization_documented':'Output image normalized to the same peak brightness as input; no disable control in inspected IAS_ImageSimulation interface.',
         'controls':{},'sampling_comparison':f['cases'].get('scene_sampling256',[])}
if not all(verified.values()):raise RuntimeError('Original dataset hash mismatch')
original=[read(sample/'b'/f'frame_{i:02d}.png') for i in range(4)]
summary['original_means']=[float(a.mean()) for a in original]
summary['original_mean_range']=float(np.ptp(summary['original_means']))
for name,rows in r['patterns'].items():
    means=[v['mean'] for v in rows];sums=[v['sum'] for v in rows]
    summary['controls'][name]={'means':means,'mean_range':float(np.ptp(means)),
                               'normalized_png_sum_relative_range':float(np.ptp(sums)/np.mean(sums))}
summary['additivity']=f['cases'].get('scene_offset',[])
for row in summary['sampling_comparison']:
    i=row['frame'];a=read(out/'followup'/f'scene_sampling256_{i:02d}.png');b=original[i]
    row['rmse_vs_original']=float(np.sqrt(np.mean((a-b)**2)))
summary['decision']='Do not accept as natural throughput variation or resume production. Preserve raw sample as quarantined evidence. Establish PSF sampling convergence and a common photometric forward model before any replacement generation; do not mix changed settings with frozen release.'
(out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
canvas=Image.new('RGB',(1024,1000),'white');d=ImageDraw.Draw(canvas)
for col in range(4):
    x=col*256
    d.text((x+8,8),f'Position {col}: original, mean={original[col].mean():.4f}',fill='black')
    canvas.paste(Image.open(sample/'b'/f'frame_{col:02d}.png').convert('RGB').resize((256,256)),(x,30))
    for row,name in enumerate(('point_center','point_offaxis')):
        path=out/f'{name}_frame_{col:02d}.png'
        if path.exists():
            y=300+row*170
            d.text((x+8,y),name+' (same PNG scale)',fill='black')
            canvas.paste(Image.open(path).convert('RGB'),(x+64,y+22))
for col,n in enumerate((128,256)):
    path=out/('psf_grids' if n==128 else f'psf_grids_{n}')/'grid_00.png'
    if path.exists():
        x=col*512;d.text((x+8,652),f'Image Simulation PSF grid, sampling={n}',fill='black')
        canvas.paste(Image.open(path).convert('RGB').resize((320,320)),(x+96,677))
canvas.save(out/'diagnostic_comparison.png')
lines=['# 0276_03 PSF / intensity scale check','',
       'Decision: keep production paused. Preserve the original sample as diagnostic evidence, not an approved physical illumination response.','',
       '| Control | Position 0 | Position 1 | Position 2 | Position 3 |',
       '|---|---:|---:|---:|---:|',
       '| Original mean | '+' | '.join(f'{v:.6f}' for v in summary['original_means'])+' |']
for name,c in summary['controls'].items():
    lines.append('| '+name+' mean | '+' | '.join(f'{v:.6f}' for v in c['means'])+' |')
lines+=['','## Findings','',
        '- Uniform controls match across positions after the simulator normalization. This does not establish equal physical throughput.',
        '- The documented peak normalization makes output intensity scene dependent. Offset additivity errors are recorded separately; PNG channel conversion and quantization must not be mistaken for absolute radiometry.',
        '- Center/off-axis bright patches are finite-patch effective responses, not unnormalized optical PSFs. Their PNG sums cannot calibrate physical energy.',
        '- The off-axis patch falls near the detector edge (centroid y around 122/128) and is clipped; do not interpret its sum changes as throughput changes.',
        '- PSF Grid exports provide PNGs only: zero scalar/RGB numerical grids and no text export. Display differences alone cannot prove convergence.',
        '- Sampling comparison changes pupil and image sampling together; it does not isolate their individual contributions.',
        '', '## Sampling sensitivity','',json.dumps(summary['sampling_comparison'],indent=2),
        '', '## Next decision','',summary['decision'],
        '', 'Before replacement: compare 128/256/512 sampling on controls and representative scenes; extract raw PSF/irradiance with a documented common scale or implement a validated linear forward operator. Freeze the revised generation version and reassess A/GT compatibility. Do not normalize each frame independently to force the brightness gate to pass.',
        '', '[Official Image Simulation documentation]('+summary['source']+')','',
        'Dataset images, frozen configuration, quality thresholds and production manifest were not edited. No training was started.']
(out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
