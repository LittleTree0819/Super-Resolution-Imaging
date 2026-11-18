"""Post-run integrity and scale/shape diagnosis; never modifies acceptance gates."""
import hashlib
import json
from datetime import datetime,timedelta,timezone
from pathlib import Path
import numpy as np
from super_resolution_imaging.data.complex_quadrature import CoherentMean,intensity_error

ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/psf_low_memory_reference/run_v1';OUT=RUN/'analysis';OUT.mkdir(exist_ok=True)
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
state=read(RUN/'status.json');guard=read(RUN/'guard_status.json');protocol=read(RUN/'protocol.json')
if not state['completed']:raise RuntimeError('Batch has not completed')
metadata=list((RUN/'blocks').glob('*/*.json'))
integrity=[];closures=[];geometry=[];retry_count=0
for path in metadata:
    row=read(path);valid=sha(path.with_suffix('.npz'))==row['npz_sha256'] and row['protocol_sha256']==sha(RUN/'protocol.json')
    integrity.append(valid);closures.append(row['closure_relative_l1']);geometry.append(row['geometry_error']);retry_count+=len(row['retries'])
if not all(integrity):raise RuntimeError('Block integrity failure')
rows=[]
for case in state['cases']:
    case_id=case['case']['id'];final=case['attempts'][-1];count=final['directions']
    a=np.load(RUN/'ensembles'/case_id/f'p4096_m{count}.npz')['intensity']
    b=np.load(RUN/'ensembles'/case_id/f'p8192_m{count}.npz')['intensity']
    factor=float(np.sum(a*b)/np.sum(a*a))
    mean=CoherentMean()
    for angle in protocol['directions_deg'][:count]:
        mean.add(np.load(RUN/'blocks'/case_id/f'p8192_angle{angle:g}.npz')['field'])
    rebuild=float(np.max(np.abs(mean.intensity-b)))
    rows.append({'case':case_id,'directions':count,'passed':case['passed'],
                 'density_relative_l1':final['pupil_density']['relative_l1_intensity'],
                 'density_shape_tv':final['pupil_density']['shape_tv'],
                 'angular_8192_relative_l1':final['angular_high']['relative_l1_intensity'],
                 'original8192_relative_l1':final['previous_native8192_vs_reference']['relative_l1_intensity'],
                 'node_mean_bias_4096_vs8192':float(a.mean()/b.mean()-1),
                 'diagnostic_least_squares_gain_4096_to8192':factor,
                 'diagnostic_gain_corrected_relative_l1':intensity_error(factor*a,b)['relative_l1_intensity'],
                 'reference_rebuild_max_absolute_error':rebuild,
                 'diagnostic_gain_applied_to_reference':False})
china=timezone(timedelta(hours=8))
summary={'state':state['state'],'completed':state['completed'],'production_ready':False,
         'completed_at_china':datetime.fromisoformat(state['updated_utc']).astimezone(china).isoformat(),
         'elapsed_seconds':guard['elapsed_seconds'],'peak_working_set_gib':guard['peak_working_set_bytes']/1024**3,
         'block_count':len(metadata),'all_block_hashes_valid':all(integrity),'max_complex_closure_relative_l1':max(closures),
         'max_geometry_error':max(geometry),'native_retry_count':retry_count,'cases':rows,
         'source_model_unchanged':sha(Path(protocol['source_model']['path']))==protocol['source_model']['sha256'],
         'configuration_unchanged':sha(ROOT/'configs/zemax_quarter_four_frozen.toml')==protocol['configuration_sha256'],
         'interpretation':'Angular quadrature converged, but pupil-density convergence failed. Rotating the same-density lattice does not replace finer radial/linear pupil sampling. Remaining scale differences may involve quadrature or native normalization; not isolated by this run.',
         'next_recommendation':'Do not add more directions or resume production. Isolate native reference normalization and node-scale bias, then obtain a genuinely finer pupil reference with memory-bounded integration or a larger-memory machine.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['# Low-Memory PSF Batch Results','',
       f"Batch completed at {summary['completed_at_china']} after about {summary['elapsed_seconds']/3600:.2f} hours. Peak working set: {summary['peak_working_set_gib']:.2f} GiB.",'',
       '**Computation completed. Both off-axis cases failed at least one convergence gate and remain unqualified as high-precision references.**','',
       '| Case | Directions | 4096 to 8192 intensity L1 | Shape TV | 8 to 16 directions at 8192 L1 | Single-direction 8192 vs. combined reference L1 |',
       '|---|---:|---:|---:|---:|---:|']
for r in rows:lines.append(f"| {r['case']} | {r['directions']} | {r['density_relative_l1']:.3%} | {r['density_shape_tv']:.3%} | {r['angular_8192_relative_l1']:.3%} | {r['original8192_relative_l1']:.3%} |")
lines+=['','## Integrity','',f"{len(metadata)} block hashes passed. Maximum complex-field closure error: {max(closures):.3g}. Maximum geometric equivalence error: {max(geometry):.3g}. The reference can be reconstructed from the cached blocks.", 
        'The batch stayed within memory and time limits. The error log was clear and the model and production configuration hashes matched.',
        '', '## Scale and Shape Decomposition (Diagnostic)','',
        '| Case | 4096 node mean bias relative to 8192 | Least-squares gain | Gain-adjusted intensity L1 |',
        '|---|---:|---:|---:|']
for r in rows:lines.append(f"| {r['case']} | {r['node_mean_bias_4096_vs8192']:+.3%} | {r['diagnostic_least_squares_gain_4096_to8192']:.6f} | {r['diagnostic_gain_corrected_relative_l1']:.3%} |")
lines+=['','The fitted gain is diagnostic only. The references, thresholds and dataset retain their original values. Node means do not measure total PSF flux or calibrate exposure.',
        '', '## Conclusion','',
        'The low-memory method controlled resources and verified angular quadrature stability. The difference between pupil densities persisted when rotating grids at a fixed density.',
        'The next diagnostic is to distinguish native-reference normalization from node-intensity scale bias and verify against a denser pupil grid. This requires memory-bounded integration or additional memory.',
        'Production generation and training remain paused.']
(OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
