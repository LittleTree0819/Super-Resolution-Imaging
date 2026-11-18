"""Explicit scope gates for converged Huygens probes and local pixel quadrature."""
import hashlib
import json
from pathlib import Path
import numpy as np
from super_resolution_imaging.zos.psf_metrics import PsfGrid,compare_grid

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/psf_reliability_v2'
base=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
status=json.loads((OUT/'confirm_status.json').read_text(encoding='utf-8'))
def grid(row):
    a=np.load(OUT/(row['key']+'.npz'))['psf']
    return PsfGrid(a,row['delta_um'],row['delta_um'],row['minx_um'],row['miny_um'])
def signature(row):return (row['position'],tuple(row['object_xy_mm']),row['wavelength_um'],row['image_n'],row['delta_um'])
grouped={}
all_rows=list(status['rows'])
high_path=OUT/'high_reference_status.json'
if high_path.exists():all_rows+=json.loads(high_path.read_text(encoding='utf-8'))['rows']
for r in all_rows:grouped.setdefault(signature(r),{})[r['pupil']]=r
checks=[]
for sig,rows in grouped.items():
    if not (4096 in rows and 8192 in rows):continue
    low,high=(8192,16384) if 16384 in rows else (4096,8192)
    first=rows[low];a=grid(first);b=grid(rows[high]);description='local patch'
    if first['delta_um']>=9.6:
        chief=first.get('chief_xy_mm')
        if chief is None:raise RuntimeError('Missing chief coordinates: cannot identify detector ROI')
        x=a.minx_um+np.arange(a.values.shape[0])*a.dx_um;y=a.miny_um+np.arange(a.values.shape[1])*a.dy_um
        mx=np.abs(x+chief[0]*1000)<=153.6+1e-6;my=np.abs(y+chief[1]*1000)<=153.6+1e-6
        if mx.sum()<5 or my.sum()<5:raise RuntimeError('Too few valid detector probe points')
        a=PsfGrid(a.values[np.ix_(mx,my)],a.dx_um,a.dy_um,float(x[mx][0]),float(y[my][0]))
        description='chief-translated physical detector ROI'
    metrics=compare_grid(a,b)
    passed=metrics['relative_l1_intensity']<=.01 and metrics['normalized_shape_tv']<=.01
    checks.append({'position':first['position'],'object_xy_mm':first['object_xy_mm'],'wavelength_um':first['wavelength_um'],
                   'scope':description,'node_count':int(a.values.size),'pupil_pair':[low,high],'passed':passed,**metrics})

def pixel_integrals(g):
    # Six 2.4-um pixels on each axis; endpoints exist in both local grids.
    bounds=np.arange(-7.2,7.2+1e-6,2.4);results=np.empty((6,6))
    steps=int(round(2.4/g.dx_um));weights=np.ones(steps+1);weights[[0,-1]]=.5
    for i in range(6):
        for j in range(6):
            x=bounds[i]+np.arange(steps+1)*g.dx_um;y=bounds[j]+np.arange(steps+1)*g.dy_um
            a=g.sample(x[:,None],y[None,:])
            results[i,j]=np.sum(a*weights[:,None]*weights[None,:])*g.dx_um*g.dy_um
    return results
patches={}
for d,n in ((.6,32),(.3,64),(.15,128)):
    stem=f'Spherical_p4096_i{n}_d{d:g}_a_x0_y0_w0.535'
    if (OUT/(stem+'.json')).exists():
        r=json.loads((OUT/(stem+'.json')).read_text(encoding='utf-8'));patches[d]=pixel_integrals(grid(r))
quad=[]
for old,new in ((.6,.3),(.3,.15)):
    if old in patches and new in patches:
        a=patches[old];b=patches[new]
        error=float(np.abs(a-b).sum()/np.abs(b).sum())
        quad.append({'delta_pair_um':[old,new],'region_um':[-7.2,7.2],'detector_pixels':[6,6],
                     'relative_l1_integrated_intensity':error,'passed':error<=.01})
pixel_pass=quad[-1]['passed'] if quad else None
sample=ROOT/'runs/div2k_quarter_four_v1/samples/0276_03'
meta=json.loads((sample/'metadata.json').read_text(encoding='utf-8'))
original_ok=all(hashlib.sha256((sample/group/f'frame_{i:02d}.png').read_bytes()).hexdigest()==sha
                for group in ('a','b') for i,sha in enumerate(meta['output_sha256'][group]))
if not original_ok:raise RuntimeError('Original sample hash mismatch')
final={'confirmation_completed':status['completed'],'completed_jobs':len(status['rows']),'planned_jobs':9,
       'production_ready':False,'dataset_modified':False,'original_sample_frames_unchanged':original_ok,
       'selected_method_for_further_validation':'Huygens Spherical; fixed method, never Auto during convergence tests',
       'pupil_checks':checks,'local_pixel_quadrature':quad,'local_pixel_quadrature_passed':pixel_pass,
       'fft_accepted_as_replacement':False,'fft_vs_huygens':base['fft_vs_huygens'],
       'all_tested_pupil_checks_passed':bool(len(checks)==4 and all(c['passed'] for c in checks)),
       'needs_pixel_refinement':bool(status['completed'] and pixel_pass is False and .15 not in patches),
       'limitations':['Sparse probes do not certify a full dense PSF or its full-energy integral',
                      'Local pixel quadrature checks only the central 6x6 detector pixels at green wavelength',
                      'Only selected lens positions, fields and wavelengths are tested',
                      'No post-truncation unit-sum normalization is permitted without separate throughput calibration'],
       'next_required':'Use measured field-specific detector ROI and preserve raw PSF scale; any failed pupil gate requires a higher reference before releasing a PSF bank.'}
guard_path=OUT/'high_reference_guard.json'
if guard_path.exists():
    final['high_reference_resource_status']=json.loads(guard_path.read_text(encoding='utf-8-sig'))
(OUT/'decision.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
lines=['# PSF Numerical Reliability and Verification','',
       '**Production generation remains paused. The diagnostics cover selected locations.**','',
       '## Observed Sources of Error','',
       '1. Auto selects Planar for the 153.6 µm window and Spherical for the 307.2 µm window. With the method fixed, the overlap differs by roughly 10⁻¹⁴.',
       '2. The original pupil sampling was too coarse. OpticStudio rejects undersampled FFT results. Switching methods requires a valid grid and verified numerical agreement.',
       '3. The original window is small compared with the roughly 4.17 mm maximum center spot radius in geometric ray tracing. This guides support selection. Diffraction energy requires its own measurement.',
       '4. Convert ZOS-API Values[x,y] to image order [y,x]. Compare physical integrals using sum times dx times dy to account for the grid spacing.',
       '', '## Corrective Strategy','',
       '- Fix the method to Huygens Spherical. Check pupil sampling, image-plane spacing and physical support independently.',
       '- Use chief-ray positions to identify the required detector support. Retain the native relative intensity scale for cropped kernels.',
       '- Screen high-density references at selected nodes, then check pixel-area integration separately. Use sparse nodes for diagnostics only.',
       '', '## High-Density Reference Checks','',
       '| Position | Object field (mm) | Wavelength (µm) | Pupil sampling pair | Nodes | Relative intensity L1 | Shape TV | 1% gate |',
       '|---|---|---:|---|---:|---:|---:|---|']
for c in checks:
    lines.append(f"| {c['position']} | {c['object_xy_mm']} | {c['wavelength_um']} | {c['pupil_pair']} | {c['node_count']} | {c['relative_l1_intensity']:.3%} | {c['normalized_shape_tv']:.3%} | {'Pass' if c['passed'] else 'Fail'} |")
lines+=['','## Image-Plane Spacing Check','', '| Spacing (µm) | Relative L1 of pixel-integrated intensity | Status |','|---|---:|---|']
for q in quad:lines.append(f"| {q['delta_pair_um']} | {q['relative_l1_integrated_intensity']:.3%} | {'Pass' if q['passed'] else 'Fail'} |")
lines+=['', 'The comparison uses the central 6 × 6 detector pixels of width 2.4 µm and two-dimensional trapezoidal integration on common physical boundaries. The result is limited to this region.',
        '', '## Findings and Scope','',
        f"Confirmed batches: {len(status['rows'])}/9. Completion flag: {status['completed']}.",
        'High-density evidence is available for the central region. Full validation requires meeting the thresholds for displaced, off-axis and wavelength cases and for pixel integration. Cases that failed remain unqualified.',
        'FFT stabilized in its own sequence, but its comparison against spherical Huygens failed. Spherical Huygens remains the reference method.',
        'The finite source and detector require only part of the PSF support. A future library should cover the effective region with explicit coordinates and a flux scale.',
        'The diagnostics retain the optical prescription, lens positions, source images and training configuration. No new dataset was generated.',
        '', '[Huygens PSF documentation](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v261/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Huygens_PSF.html)',
        '', '[ZOS-API grid coordinate documentation](https://ansyshelp.ansys.com/public/Views/Secured/Zemax/v251/en/OpticStudio_User_Guide/OpticStudio_Help/topics/Putting_DataGrids_all_together.html)']
if 'high_reference_resource_status' in final:
    resource=final['high_reference_resource_status']
    lines+=['','## Higher-Density Reference and Resource Limits','',
            f"16384-level status: {resource['state']}. Working set limit: 18 GiB. Minimum available memory: 3 GiB.",
            'A resource stop leaves convergence unverified. Off-axis cases without a higher-density reference remain unqualified.']
(OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(final,indent=2))
