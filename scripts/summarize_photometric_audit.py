"""Evaluate diagnostic gates; never declares a failed forward model frozen."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/photometric_forward_audit_v1'
protocol_path=ROOT/'configs/photometric_forward_audit_protocol.json'
protocol=json.loads(protocol_path.read_text(encoding='utf-8'))
raw=json.loads((OUT/'raw_psf/report.json').read_text(encoding='utf-8'))
images=json.loads((OUT/'image_sampling/report.json').read_text(encoding='utf-8'))
linear=json.loads((OUT/'linear_forward/report.json').read_text(encoding='utf-8'))
def read(path):return np.asarray(Image.open(path).convert('L'),dtype=float)/255
raw_rows=[]
for position in ('a','b0','b1'):
    for field in ('center','edge','opposite'):
        prev=None
        for n in (128,256,512):
            key=f'{position}_{field}_p{n}_i128_w0.535'
            a=np.load(OUT/'raw_psf'/(key+'.npz'))['psf']
            row={'position':position,'field':field,'sampling':n,'relative_integral':float(a.sum())}
            if prev is not None:
                row['shape_total_variation_vs_previous']=float(np.abs(a/a.sum()-prev/prev.sum()).sum()/2)
                row['integral_relative_change_vs_previous']=float(abs(a.sum()/prev.sum()-1))
            raw_rows.append(row);prev=a
small=np.load(OUT/'raw_psf/a_center_p512_i128_w0.535.npz')['psf']
large=np.load(OUT/'raw_psf/a_center_p512_i256_w0.535.npz')['psf']
support_change=float(abs(large.sum()/small.sum()-1))
central=large[64:192,64:192]
tol=protocol['engineering_acceptance_tolerances']
psf512=[r for r in raw_rows if r['sampling']==512]
image512=[r for r in images['rows'] if r['sampling']==512 and not r['ideal']]
gates={
    'raw_psf_integral_convergence':all(r['integral_relative_change_vs_previous']<=tol['psf_integral_relative_change_256_to_512_max'] for r in psf512),
    'raw_psf_shape_convergence':all(r['shape_total_variation_vs_previous']<=tol['psf_shape_total_variation_256_to_512_max'] for r in psf512),
    'raw_psf_window_support':support_change<=tol['psf_support_relative_integral_change_max'],
    'image_convergence':False if any(r['mae_vs_previous']>tol['image_mae_256_to_512_max'] for r in image512) else (True if len(image512)==3 else None),
    'linear_float_implementation':all(r['additivity_max_error']<tol['linear_float_additivity_max_absolute_error'] for r in linear['positions']),
    'full_forward_physics_validated':False,
}
sample=ROOT/'runs/div2k_quarter_four_v1/samples/0276_03'
meta=json.loads((sample/'metadata.json').read_text(encoding='utf-8'))
hashes={}
for group in ('a','b'):
    for i,expected in enumerate(meta['output_sha256'][group]):
        path=sample/group/f'frame_{i:02d}.png'
        hashes[f'{group}/{path.name}']=hashlib.sha256(path.read_bytes()).hexdigest()==expected
hashes['target.png']=hashlib.sha256((sample/'target.png').read_bytes()).hexdigest()==meta['target_sha256']
if not all(hashes.values()):raise RuntimeError('Original sample changed')
compatibility=[]
for group,old in [('a',sample/'a/frame_00.png'),('gt',sample/'target.png')]:
    original=read(old)
    for n in (128,256,512):
        path=OUT/'image_sampling'/f'{group}_{n}.png'
        if path.exists():
            a=read(path);compatibility.append({'group':group,'sampling':n,'mae_vs_original':float(np.abs(a-original).mean()),'mean':float(a.mean())})
decision={
    'diagnostics_completed':bool(raw['completed'] and images['completed'] and linear['completed']),
    'production_settings_frozen':False,'generation_resumed':False,'training_started':False,
    'release_decision':'HOLD: PSF sampling and finite support failed; fixed-scale code passes but physics is not validated.',
    'gates':gates,'protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
    'raw_psf_rows':raw_rows,'support_relative_integral_change':support_change,
    'large_window_common_crop_mae_relative_to_small_peak':float(np.abs(central-small).mean()/small.max()),
    'image_sampling_rows':images['rows'],'A_GT_compatibility':compatibility,
    'original_sample_hash_checks':hashes,
    'replacement_scope':protocol['replacement_scope_if_forward_model_changes'],
    'next_required':['Hold physical PSF window fixed and extend pupil sampling adaptively beyond 512; stop only at convergence',
                     'Enlarge physical PSF support independently and verify captured energy and central-grid invariance',
                     'Validate spatially varying RGB PSFs, wavelength-specific geometry and relative illumination separately',
                     'Only after all physical gates pass freeze a new version; regenerate A/B/GT consistently; do not reuse old A checkpoint by default']}
(OUT/'decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
canvas=Image.new('RGB',(900,900),'white');draw=ImageDraw.Draw(canvas)
arrays=[np.load(OUT/'raw_psf'/f'{p}_center_p{n}_i128_w0.535.npz')['kernel'] for p in ('a','b0','b1') for n in (128,256,512)]
ref=max(float(a.max()) for a in arrays)
for i,a in enumerate(arrays):
    row,col=divmod(i,3);x=col*300;y=row*300
    draw.text((x+8,y+8),f'{("A","B0","B1")[row]} pupil={(128,256,512)[col]}',fill='black')
    display=np.clip((np.log10(np.maximum(a/ref,1e-4))+4)/4,0,1)
    canvas.paste(Image.fromarray(np.uint8(display*255)).convert('RGB').resize((256,256)),(x+22,y+30))
canvas.save(OUT/'psf_convergence.png')
lines=['# PSF Convergence and Common Intensity Scale Audit','',
       '**Production settings remain provisional. Full generation and training remain paused.**','',
       '## 1. Sampling Convergence','',
       'Fixed physical grid of 128 × 128 with 1.2 µm spacing and unit-peak normalization disabled. Integrals are relative to the Strehl reference.','',
       '| Position | Field | Pupil sampling | Relative integral | Shape TV vs. previous level | Integral change vs. previous level |',
       '|---|---|---:|---:|---:|---:|']
for r in raw_rows:
    lines.append(f"| {r['position']} | {r['field']} | {r['sampling']} | {r['relative_integral']:.6f} | "+
                 (f"{r['shape_total_variation_vs_previous']:.4f} | {r['integral_relative_change_vs_previous']:.2%} |" if 'shape_total_variation_vs_previous' in r else '— | — |'))
lines+=['',f'Doubling the central PSF window width changed its integral by {support_change:.2%}. Truncation and numerical error can contribute to this difference.',
        '', '### Full-Image Simulation','', '| Group | Sampling | Mean | MAE vs. previous level |','|---|---:|---:|---:|']
for r in images['rows']:
    lines.append(f"| {r['group']} | {r['sampling']} | {r['mean']:.6f} | "+(f"{r['mae_vs_previous']:.6f}" if 'mae_vs_previous' in r else '—')+' |')
lines+=['','## 2. Common Intensity Scale','',
        'The floating-point operator uses a fixed intensity scale. Quantization and clipping apply only to previews.',
        'Three unit tests covered energy, orientation, boundaries and linearity. Maximum additivity error in the simulated PSF scene: '+f"{max(r['additivity_max_error'] for r in linear['positions']):.3g}.", 
        '', '**Physical limitations: ** The diagnostic uses a central PSF, assumed unit flux and monochromatic geometry. Field variation, chromatic geometry, relative illumination and PSF convergence remain unresolved.',
        '', '## 3. Release and Recalculation Decision','',
        'The diagnostic protocol and failure thresholds are fixed. Production generation requires further validation before a move from 128 to 512 samples.',
        'If the intensity model changes, regenerate centered A, all four B frames and the ideal target together. Preserve source crops, sample IDs, source-level splits, lens positions, dimensions and the exclusion list.',
        'Retrain Stage A for the new version or validate weight transfer in a separate controlled experiment. Keep generated versions separate.',
        '', 'A and target comparison','', '| Group | New sampling | MAE vs. original image |','|---|---:|---:|']
for r in compatibility:lines.append(f"| {r['group']} | {r['sampling']} | {r['mae_vs_original']:.6f} |")
lines+=['','## Remaining Validation Checks','',
        '1. Increase pupil sampling at fixed physical support and extend to 1024 or higher as needed. Determine the convergence level from the measurements.',
        '2. Increase PSF support independently. Check integral stability and overlap agreement before calibrating the energy scale.',
        '3. Validate RGB, multiple fields, chromatic geometry and relative illumination before freezing a production version.',
        '',f"Diagnostic completion status: {decision['diagnostics_completed']}. All original A, B and target hashes matched. The source dataset and production configuration retain their original values.", 
        '', '[Huygens PSF documentation](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v251/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Huygens_PSF.html)',
        '', '[Image Simulation documentation](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v251/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Image_Simulation.html)']
(OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'completed':decision['diagnostics_completed'],'gates':gates,'compatibility':compatibility},indent=2))
