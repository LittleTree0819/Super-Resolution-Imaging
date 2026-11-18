"""Resumable coherent quadrature reference with one native pupil grid resident.

The union of rotated pupil grids is NOT a uniform 16384 lattice. Accept only
after independent pupil-density and direction-count convergence checks. No
scene, GT, training metric, per-block gain fit, or phase fit is used.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from super_resolution_imaging.config import load_config
from super_resolution_imaging.zos.connection import connect
from super_resolution_imaging.zos.model import validate_source
from super_resolution_imaging.zos.rotated_pupil import apply_equivalent_rotation,rotation,trace_probe,native_complex_roi
from super_resolution_imaging.data.complex_quadrature import CoherentMean,dyadic_angles,intensity_error
from super_resolution_imaging.zos.psf_metrics import PsfGrid

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'runs/psf_low_memory_reference/run_v1')
parser.add_argument('--max-directions',type=int,default=16);args=parser.parse_args()
OUT=args.output.resolve();OUT.relative_to((ROOT/'runs').resolve());OUT.mkdir(parents=True,exist_ok=True)
angles=dyadic_angles(args.max_directions)
if args.max_directions<4:raise ValueError('At least four directions are required')
cfg_path=ROOT/'configs/zemax_quarter_four_frozen.toml';cfg=load_config(cfg_path);source=validate_source(cfg)
prepared=ROOT/'runs/div2k_quarter_four_v1/prepared_model.zmx'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)
probe=json.loads((OUT.parent/'probe_v3/report.json').read_text(encoding='utf-8'))
if not (probe['completed'] and probe['geometry_passed'] and probe['field_closure_passed'] and probe['independent_quadrature_observed'] and probe['field_90_vs_0_relative_l2']<1e-5):
    raise RuntimeError('Equivalent-geometry / complex-field probe not validated')
protocol={'algorithm':'equivalent_rotated_pupil_coherent_quadrature_v1','source_model':source,
          'prepared_model_sha256':sha(prepared),'configuration_sha256':sha(cfg_path),
          'physics_module_sha256':sha(ROOT/'src/super_resolution_imaging/zos/rotated_pupil.py'),
          'aggregation_module_sha256':sha(ROOT/'src/super_resolution_imaging/data/complex_quadrature.py'),
          'zosapi_assembly_sha256':sha(cfg.zemax.install_dir/'ZOSAPI.dll'),
          'interfaces_assembly_sha256':sha(cfg.zemax.install_dir/'ZOSAPI_Interfaces.dll'),
          'probe_sha256':sha(OUT.parent/'probe_v3/report.json'),
          'pupil_levels':[4096,8192],'directions_deg':angles,'image_n':32,'delta_um':38.4,
          'integration_method':'Spherical','normalization':False,'use_polarization':False,
          'threshold_relative_l1':.01,'threshold_shape_tv':.01,
          'control_threshold':.01,'closure_threshold':1e-8,
          'no_phase_or_gain_fitting':True,'no_intensity_averaging':True,
          'scope':'Reference values at physical detector nodes only; not a dense PSF bank',
          'nominal_sample_count_warning':'M*N^2 counts quadrature evaluations; it is not a uniform finer Cartesian grid',
          'cases':[{'id':'blue_b0_edge','position':0,'field':[2.1375,2.1375],'wave':.465},
                   {'id':'red_b2_edge','position':2,'field':[-2.1375,-2.1375],'wave':.606}]}
plan_path=OUT/'protocol.json'
if plan_path.exists() and json.loads(plan_path.read_text(encoding='utf-8'))!=protocol:
    raise RuntimeError('Protocol or physics code changed; use a new output version')
if not plan_path.exists():atomic_json(plan_path,protocol)
protocol_sha=sha(plan_path)
state={'state':'starting','worker_pid':os.getpid(),'completed':False,'production_ready':False,
       'generation_resumed':False,'training_started':False,'protocol_sha256':protocol_sha,
       'completed_blocks':0,'cases':[],'started_utc':datetime.now(timezone.utc).isoformat()}
seen_blocks=set()
def count_block(path):
    seen_blocks.add(str(path));state['completed_blocks']=len(seen_blocks)
def save():
    state['updated_utc']=datetime.now(timezone.utc).isoformat();atomic_json(OUT/'status.json',state)
def pause_check():
    if (OUT/'pause.request').exists():raise InterruptedError('Pause requested at a safe calculation boundary')
def refresh_report():
    subprocess.run([sys.executable,str(ROOT/'scripts/write_low_memory_psf_report.py'),'--output',str(OUT)],cwd=ROOT,check=True)
def acceptable(error):return error['relative_l1_intensity']<=.01 and error['shape_tv']<=.01
base_folder=ROOT/'runs/psf_reliability_v2'

def block(case,pupil,angle):
    pause_check();folder=OUT/'blocks'/case['id'];folder.mkdir(parents=True,exist_ok=True)
    stem=f'p{pupil}_angle{angle:g}';target=folder/(stem+'.npz');meta_path=folder/(stem+'.json')
    if target.exists() and meta_path.exists():
        meta=json.loads(meta_path.read_text(encoding='utf-8'))
        if meta['protocol_sha256']!=protocol_sha or sha(target)!=meta['npz_sha256']:raise RuntimeError('Invalid cached block')
        if meta['case']!=case or meta['pupil']!=pupil or meta['angle_deg']!=angle:raise RuntimeError('Cached case mismatch')
        data=np.load(target);field=data['field'].copy();linear=data['linear'].copy()
        if not np.isfinite(field).all() or np.abs(np.abs(field)**2-linear).sum()/linear.sum()>1e-8:raise RuntimeError('Cached field closure failed')
        count_block(target);save();return field,linear,meta
    position=(0.,0.) if case['position'] is None else cfg.dataset.positions_mm[case['position']]
    field_xy=np.asarray(case['field']);pupils=np.array([[0,0],[.2,.3],[-.7,.2],[.4,-.5],[.9,.1],[.1,.9]])
    state.update(current_case=case['id'],current_pupil=pupil,current_angle_deg=angle,current_component='connecting');save()
    started=time.monotonic();errors=[]
    for attempt in range(1,4):
        try:
            with connect(cfg.zemax.install_dir) as session:
                session.system.LoadFile(str(prepared),False)
                apply_equivalent_rotation(session,cfg,position,field_xy,case['wave'],0.)
                reference=trace_probe(session,field_xy,pupils@rotation(-angle).T)
                apply_equivalent_rotation(session,cfg,position,field_xy,case['wave'],angle)
                traced=trace_probe(session,rotation(angle)@field_xy,pupils)
                geometry_error=float(np.max(np.abs(reference[:,2:8]-traced[:,2:8])))
                optical_path_error=float(np.max(np.abs(reference[:,8]-traced[:,8])))
                intensity_error_probe=float(np.max(np.abs(reference[:,9]-traced[:,9])))
                if not np.array_equal(reference[:,:2],traced[:,:2]) or geometry_error>1e-8 or optical_path_error>1e-8 or intensity_error_probe>1e-10:
                    raise RuntimeError('Rigid-rotation geometry failed')
                chief=traced[0,2:4].copy()
                def progress(kind):
                    pause_check();state['current_component']=kind;save()
                e,linear,meta=native_complex_roi(session,pupil,32,38.4,chief,progress=progress)
            break
        except InterruptedError:raise
        except Exception as exc:
            errors.append({'attempt':attempt,'error':str(exc)});state['last_retries']=errors;save()
            if attempt==3:raise
    temp=target.with_suffix('.tmp')
    with temp.open('wb') as handle:np.savez_compressed(handle,field=e,linear=linear)
    temp.replace(target)
    meta.update(protocol_sha256=protocol_sha,npz_sha256=sha(target),case=case,pupil=pupil,angle_deg=angle,
                chief_xy_mm=chief.tolist(),geometry_error=geometry_error,optical_path_error=optical_path_error,
                ray_intensity_error=intensity_error_probe,elapsed_s=time.monotonic()-started,retries=errors)
    atomic_json(meta_path,meta);count_block(target);save()
    print(json.dumps({'block_completed':str(target),'elapsed_s':meta['elapsed_s'],'closure':meta['closure_relative_l1']}),flush=True)
    return e,linear,meta

def ensemble(case,pupil,count):
    mean=CoherentMean();grid=None;single=None
    for angle in angles[:count]:
        e,lin,meta=block(case,pupil,angle)
        if grid is not None and grid!=meta['grid']:raise RuntimeError('Coordinate grids differ across rotations')
        grid=meta['grid'];mean.add(e)
        if angle==0.:single=lin
    folder=OUT/'ensembles'/case['id'];folder.mkdir(parents=True,exist_ok=True)
    target=folder/f'p{pupil}_m{count}.npz'
    np.savez_compressed(target,field=mean.field,intensity=mean.intensity,single=single,
                        x_um=np.asarray(grid['x_um']),y_um=np.asarray(grid['y_um']))
    return mean.intensity,grid,single

def old_reference(case,grid):
    pos='a' if case['position'] is None else 'b'+str(case['position'])
    x,y=case['field'];w=case['wave']
    stem=f'Spherical_p8192_i32_d38.4_{pos}_x{x:g}_y{y:g}_w{w:g}'
    meta=json.loads((base_folder/(stem+'.json')).read_text(encoding='utf-8'))
    data=np.load(base_folder/(stem+'.npz'))['psf']
    g=PsfGrid(data,meta['delta_um'],meta['delta_um'],meta['minx_um'],meta['miny_um'])
    return g.sample(np.asarray(grid['x_um'])[:,None],np.asarray(grid['y_um'])[None,:])

def run():
    save();state['state']='validating_control';save()
    control={'id':'center_control','position':None,'field':[0.,0.],'wave':.535}
    control_ok=False
    for pupil in (4096,8192):
        intensity,grid,_=ensemble(control,pupil,4)
        error=intensity_error(intensity,old_reference(control,grid))
        state['control']={'pupil':pupil,'directions':4,'error_against_converged_native':error,'passed':acceptable(error)};save();refresh_report()
        if acceptable(error):control_ok=True;break
    if not control_ok:raise RuntimeError('Low-memory method failed the independent native control')
    for case in protocol['cases']:
        state['state']='running_reference';save();attempts=[];passed=False
        count=4
        while count<=args.max_directions:
            low,grid,_=ensemble(case,4096,count);high,grid_high,single=ensemble(case,8192,count)
            if grid!=grid_high:raise RuntimeError('Pupil-density comparison grids differ')
            low_previous,_,_=ensemble(case,4096,count//2);high_previous,_,_=ensemble(case,8192,count//2)
            density=intensity_error(low,high)
            angular_low=intensity_error(low_previous,low);angular_high=intensity_error(high_previous,high)
            old=old_reference(case,grid)
            record={'directions':count,'pupil_density':density,'angular_low':angular_low,'angular_high':angular_high,
                    'previous_native8192_vs_reference':intensity_error(old,high),
                    'nominal_upper_quadrature_evaluations':count*8192**2,
                    'passed':all(acceptable(v) for v in (density,angular_low,angular_high))}
            attempts.append(record);state['active_case_checks']={'case':case['id'],'attempts':attempts};save();refresh_report()
            print(json.dumps(state['active_case_checks']),flush=True)
            if record['passed']:passed=True;break
            count*=2
        result={'case':case,'passed':passed,'attempts':attempts,
                'reference_file':str(OUT/'ensembles'/case['id']/f'p8192_m{count if passed else args.max_directions}.npz'),
                'sparse_reference_only':True}
        atomic_json(OUT/(case['id']+'_result.json'),result);state['cases'].append(result);save();refresh_report()
    validate_source(cfg)
    state['all_reference_cases_passed']=all(c['passed'] for c in state['cases'])
    state['state']='completed' if state['all_reference_cases_passed'] else 'completed_unconverged'
    state['completed']=True;state['current_component']='none';save();refresh_report()

try:run()
except InterruptedError as exc:
    state.update(state='paused',error=str(exc));save();refresh_report()
except Exception as exc:
    state.update(state='failed',error=str(exc));save();refresh_report();raise
