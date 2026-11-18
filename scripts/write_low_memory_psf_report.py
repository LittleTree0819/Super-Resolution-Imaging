"""Standalone status/report writer, including interrupted guarded runs."""
import json
import argparse
from pathlib import Path

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=root/'runs/psf_low_memory_reference/run_v1');args=parser.parse_args()
out=args.output.resolve();out.relative_to((root/'runs').resolve());out.mkdir(parents=True,exist_ok=True)
def read(path):return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
state=read(out/'status.json');guard=read(out/'guard_status.json');protocol=read(out/'protocol.json')
effective=guard.get('state') if guard.get('state') in ('memory_limited','time_budget_reached','guard_failed','worker_failed') else state.get('state','not_started')
lines=['# Low-Memory High-Density Off-Axis PSF Reference','',f'Current status: **{effective}**.','',
       '## Method and Validation Scope','',
       'Rotate coordinates to change pupil grid orientation in an equivalent system. Run native spherical Huygens in batches and integrate stored complex amplitudes coherently. Keep one 4096 or 8192 pupil grid in memory at a time.',
       'The result combines quadrature on rotated grids. A separate convergence check is needed to compare it with a regular 16384 × 16384 grid.',
       'The method integrates complex amplitudes without intensity averaging, fitted phase or gain, or output normalization. It checks geometric equivalence and complex-field closure for each block.',
       'Native Normalize=False values remain relative to the ideal PSF. Absolute irradiance requires separate flux calibration.',
       'Use the nodes needed by the detector. Exclude anomalous peripheral export cells and reject non-finite values or failed closure inside the region of interest.',
       '', '## Independent Control','',json.dumps(state.get('control',{}),indent=2,ensure_ascii=False),
       '', '## Two Off-Axis Cases','',
       '| Case | Directions | Pupil-density relative L1 | Lower-density angular refinement L1 | Higher-density angular refinement L1 | Status |',
       '|---|---:|---:|---:|---:|---|']
cases=state.get('cases',[])
for case_id in ('blue_b0_edge','red_b2_edge'):
    result=read(out/(case_id+'_result.json'))
    attempts=result.get('attempts',[])
    active=state.get('active_case_checks',{})
    if not attempts and active.get('case')==case_id:attempts=active.get('attempts',[])
    for a in attempts:
        lines.append(f"| {case_id} | {a['directions']} | {a['pupil_density']['relative_l1_intensity']:.3%} | {a['angular_low']['relative_l1_intensity']:.3%} | {a['angular_high']['relative_l1_intensity']:.3%} | {'Pass' if a['passed'] else 'Fail'} |")
    if not attempts:lines.append(f'| {case_id} | — | — | — | — | Pending |')
lines+=['','Require relative intensity L1 and shape TV to be at most 1%. Check pupil refinement from 4096 to 8192 and doubling of the direction count. Use optical diagnostics to select settings.',
        '', '## Progress and Resources','',
        f"Snapshot time (UTC): {state.get('updated_utc','—')}. See status.json for current progress.",
        f"Independent blocks read or completed: {state.get('completed_blocks',0)}.",
        f"Current case, sampling level, angle and component: {state.get('current_case','—')} / {state.get('current_pupil','—')} / {state.get('current_angle_deg','—')} / {state.get('current_component','—')}.",
        f"Worker PID: {guard.get('worker_pid',state.get('worker_pid','—'))}. Guardian PID: {guard.get('guardian_pid','—')}.",
        'Resource limits are an 18 GiB process working set, at least 3 GiB available physical memory and a 12-hour runtime. A resource stop leaves convergence unverified.',
        '', '## Resume and Pause','',
        'Rerun scripts/run_low_memory_psf_reference.ps1 to verify protocol and cache hashes and reuse completed blocks. Run one instance at a time.',
        'Create pause.request in the run directory to pause at the next computation boundary. Once the process exits, remove the request file and rerun the script to resume.',
        '', '## Limitations','',
        'These reference nodes cover selected off-axis locations. Production requires a full-field PSF library and independent intensity calibration.',
        'This diagnostic leaves the source model and production dataset intact. Generation and training remain paused.']
if state.get('error'):lines+=['','Error: '+state['error']]
if guard.get('error'):lines+=['','Guardian error: '+guard['error']]
(out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
decision={'state':effective,'completed':state.get('completed',False),
          'all_reference_cases_passed':state.get('all_reference_cases_passed',False),
          'production_ready':False,'generation_resumed':False,'training_started':False,
          'scope':'Sparse off-axis reference nodes, not a production PSF bank','cases':cases,'guard':guard}
(out/'decision.json').write_text(json.dumps(decision,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'state':effective,'completed':decision['completed'],'all_reference_cases_passed':decision['all_reference_cases_passed']}))
