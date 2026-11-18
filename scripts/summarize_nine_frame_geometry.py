"""Summarize ray geometry on the physical LR sensor and full object field."""
import json
from pathlib import Path
import numpy as np
from analyze_nine_frame_shifts import TARGET_SHIFTS
from super_resolution_imaging.config import load_config
from super_resolution_imaging.zos.model import validate_source

for name,config in [('nine_frame_geometry_fresh','configs/zemax_9frame_pilot_refined.toml'),('nine_frame_geometry_symmetric','configs/zemax_9frame_geometric.toml')]:
    root=Path('runs')/name
    report=json.loads((root/'geometry.json').read_text())
    cfg=load_config(Path(config))
    report['metadata']['source_model']=validate_source(cfg)
    report['metadata']['positions_mm']=[list(p) for p in cfg.dataset.positions_mm]
    (root/'geometry.json').write_text(json.dumps(report,indent=2)+'\n')
    rows=report['rows']
    sensor_half=cfg.simulation.lr_pixels*cfg.simulation.lr_pixel_size_mm/2
    fine=[r for r in rows if r['pupil_sampling']==33]
    sensor=[r for r in fine if max(abs(x) for x in r['center_chief_xy_mm'])<=sensor_half]
    lookup={(r['pupil_sampling'],r['frame'],tuple(r['object_xy_mm'])):r for r in rows}
    by_frame={}
    for frame in range(9):
        selected=[r for r in sensor if r['frame']==frame]
        chiefs=np.array([r['chief_shift_png_lr_px'] for r in selected])
        centroids=np.array([r['centroid_shift_png_lr_px'] for r in selected])
        by_frame[str(frame)]={'chief_mean_png_lr_px':chiefs.mean(0).tolist(),'chief_range_png_lr_px':{'min':chiefs.min(0).tolist(),'max':chiefs.max(0).tolist()},'max_target_error_lr_px':float(np.max(np.linalg.norm(chiefs-np.array(TARGET_SHIFTS[frame]),axis=1))),'centroid_mean_png_lr_px':centroids.mean(0).tolist(),'max_ray_bundle_displacement_spread_lr_px':max(r['pupil_displacement_spread_lr_px'] for r in selected)}
    summary={'source_model':report['metadata']['source_model'],'config':config,'sensor_half_width_mm':sensor_half,'object_grid_points':len(fine)//9,'sensor_covered_grid_points':len(sensor)//9,'all_chiefs_valid':all(r['chief_valid'] for r in fine),'frames':by_frame,'max_sensor_target_error_lr_px':max(v['max_target_error_lr_px'] for v in by_frame.values()),'max_centroid_sampling_change_lr_px':max(float(np.linalg.norm(np.array(r['centroid_shift_png_lr_px'])-lookup[(17,r['frame'],tuple(r['object_xy_mm']))]['centroid_shift_png_lr_px'])) for r in fine),'max_relative_rms_radius_change':max(abs(r['rms_radius_change_lr_px'])/r['rms_spot_radius_center_lr_px'] for r in fine),'limitations':['Chief rays describe geometric mapping, not diffraction image centroid.','Uniform full-pupil centroids include rays outside the small sensor; do not use as PNG registration targets.','Image sampling, PSF truncation, illumination and quantization are not modeled by geometric rays.','Sensor coverage is based on center-position chief intersection; finite pixel integration not modeled.']}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    if name=='nine_frame_geometry_fresh':
        shift_report={'frames':{str(i):{'median_measured_shift_xy':by_frame[str(i)]['chief_mean_png_lr_px']} for i in range(9)}}
        (root/'chief_shift_report.json').write_text(json.dumps(shift_report,indent=2)+'\n')
    print(name, 'max_sensor_error',summary['max_sensor_target_error_lr_px'])
