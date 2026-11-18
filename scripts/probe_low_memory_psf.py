"""Test physical equivalence and scalar field closure before high-density use."""
import json
from pathlib import Path
import numpy as np
from super_resolution_imaging.config import load_config
from super_resolution_imaging.zos.connection import connect
from super_resolution_imaging.zos.model import validate_source
from super_resolution_imaging.zos.rotated_pupil import rotation,apply_equivalent_rotation,native_grid,trace_probe

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/psf_low_memory_reference/probe_v3'
OUT.mkdir(parents=True,exist_ok=True)
cfg=load_config(ROOT/'configs/zemax_quarter_four_frozen.toml');validate_source(cfg)
position=cfg.dataset.positions_mm[0];field=np.array([2.1375,2.1375]);wave=.465
angles=(0.,22.5,45.,67.5,90.)
pupils=np.array([[0,0],[.2,.3],[-.7,.2],[.4,-.5],[.9,.1],[.1,.9],[-.3,-.8]])
report={'completed':False,'production_ready':False,'geometry':[],'fields':[]}
def save():(OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
with connect(cfg.zemax.install_dir) as session:
    session.system.LoadFile(str(ROOT/'runs/div2k_quarter_four_v1/prepared_model.zmx'),False)
    apply_equivalent_rotation(session,cfg,position,field,wave,0.)
    base=trace_probe(session,field,pupils)
    references={theta:trace_probe(session,field,pupils@rotation(-theta).T) for theta in angles}
    for theta in angles:
        apply_equivalent_rotation(session,cfg,position,field,wave,theta)
        current=trace_probe(session,rotation(theta)@field,pupils)
        ref=references[theta]
        same_errors=bool(np.array_equal(current[:,:2],ref[:,:2]))
        err=float(np.max(np.abs(current[:,2:8]-ref[:,2:8])))
        report['geometry'].append({'angle_deg':theta,'same_errors':same_errors,'max_position_direction_error':err,
                                   'difference_vs_unrotated_pupil_xy_mm':float(np.max(np.abs(current[:,2:4]-base[:,2:4])))})
        save();print('geometry',report['geometry'][-1],flush=True)
    report['geometry_passed']=all(r['same_errors'] and r['max_position_direction_error']<1e-8 for r in report['geometry'])
    if report['geometry_passed']:
        for theta in angles:
            apply_equivalent_rotation(session,cfg,position,field,wave,theta)
            lin,meta=native_grid(session,256,32,38.4,'Linear',strict_finite=False)
            re,_=native_grid(session,256,32,38.4,'Real',strict_finite=False);im,_=native_grid(session,256,32,38.4,'Imaginary',strict_finite=False)
            xs=meta['minx_um']+np.arange(lin.shape[0])*meta['dx_um'];ys=meta['miny_um']+np.arange(lin.shape[1])*meta['dy_um']
            mx=np.abs(xs+base[0,2]*1000)<=153.6;my=np.abs(ys+base[0,3]*1000)<=153.6
            lin=lin[np.ix_(mx,my)];re=re[np.ix_(mx,my)];im=im[np.ix_(mx,my)]
            if not all(np.isfinite(v).all() for v in (lin,re,im)):raise RuntimeError('Invalid required ROI field')
            e=re+1j*im;closure=float(np.abs(np.abs(e)**2-lin).sum()/lin.sum())
            np.savez_compressed(OUT/f'angle_{theta:g}.npz',linear=lin,field=e)
            row={'angle_deg':theta,'closure_relative_l1':closure,'linear_sum':float(lin.sum()),'grid':meta}
            report['fields'].append(row);save();print('field',row,flush=True)
        a=np.load(OUT/'angle_0.npz');b=np.load(OUT/'angle_90.npz');c=np.load(OUT/'angle_45.npz')
        report['field_90_vs_0_relative_l2']=float(np.linalg.norm(a['field']-b['field'])/np.linalg.norm(a['field']))
        phase=np.vdot(b['field'],a['field']);phase=phase/abs(phase)
        report['field_90_vs_0_phase_aligned_relative_l2']=float(np.linalg.norm(a['field']-b['field']*phase)/np.linalg.norm(a['field']))
        report['linear_45_vs_0_relative_l1']=float(np.abs(a['linear']-c['linear']).sum()/a['linear'].sum())
        report['field_closure_passed']=all(r['closure_relative_l1']<1e-6 for r in report['fields'])
        report['independent_quadrature_observed']=report['linear_45_vs_0_relative_l1']>1e-6
report['completed']=True;save();print(json.dumps(report,indent=2),flush=True)
