"""Trace field/pupil grids to distinguish geometric shift from image registration."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from super_resolution_imaging.config import load_config
from super_resolution_imaging.zos.connection import connect
from super_resolution_imaging.zos.model import set_lens_shift, validate_source


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=Path('configs/zemax_9frame_pilot_refined.toml'))
    parser.add_argument('--output',type=Path,default=Path('runs/nine_frame_geometry'))
    parser.add_argument('--probe',action='store_true')
    args=parser.parse_args(); cfg=load_config(args.config)
    with connect(cfg.zemax.install_dir) as session:
        source_metadata=validate_source(cfg)
        session.system.LoadFile(str(cfg.zemax.source_model.resolve()),False)
        fields=session.system.SystemData.Fields
        waves=session.system.SystemData.Wavelengths
        metadata={'source_model':source_metadata,'positions_mm':[list(p) for p in cfg.dataset.positions_mm],'field_type':str(fields.GetFieldType()),'field_normalization':str(fields.Normalization),'fields':[{'x':float(fields.GetField(i).X),'y':float(fields.GetField(i).Y)} for i in range(1,int(fields.NumberOfFields)+1)],'wavelengths':[{'number':i,'wavelength_um':float(waves.GetWavelength(i).Wavelength),'weight':float(waves.GetWavelength(i).Weight)} for i in range(1,int(waves.NumberOfWavelengths)+1)]}
        print(json.dumps(metadata,indent=2),flush=True)
        if args.probe:
            print('fields properties',dir(fields),flush=True)
            tool=session.system.Tools.OpenBatchRayTrace()
            try:
                data=tool.CreateNormUnpol(1,session.ZOSAPI.Tools.RayTrace.RaysType.Real,int(session.system.LDE.NumberOfSurfaces)-1)
                print('AddRay',data.AddRay.__doc__,flush=True)
                print('ReadNextResult',data.ReadNextResult.__doc__,flush=True)
            finally: tool.Close()
            return
        if metadata['field_type']!='ObjectHeight' or metadata['field_normalization']!='Radial':
            raise RuntimeError('This diagnostic requires ObjectHeight fields with Radial normalization')
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(args.output)
        args.output.mkdir(parents=True,exist_ok=True)
        max_field=max(np.hypot(f['x'],f['y']) for f in metadata['fields'])
        half=cfg.simulation.field_height_mm/2
        field_xy=np.array([(x,y) for y in np.linspace(-half,half,5) for x in np.linspace(-half,half,5)])
        if int(waves.NumberOfWavelengths)!=1:
            raise RuntimeError('Select wavelength explicitly for a multi-wavelength model')
        primary=1
        metadata.update({'normalization_radius_mm':max_field,'object_grid_xy_mm':field_xy.tolist(),'primary_wavelength_number':primary,'lr_pixel_size_mm':cfg.simulation.lr_pixel_size_mm,'image_surface':int(session.system.LDE.NumberOfSurfaces)-1,'coordinates':'ray x,y are physical image-plane coordinates; PNG y sign is inverted; PNG x uses same sign','centroid_definition':'uniform normalized-pupil geometric centroid; valid and unvignetted rays shared with center lens position; not diffraction PSF centroid'})
        import System
        opd=System.Enum.Parse(session.ZOSAPI.Tools.RayTrace.OPDMode,'None')
        rows=[]
        for pupil_n in (17,33):
            pupils=[(0.,0.)]+[(float(x),float(y)) for y in np.linspace(-1,1,pupil_n) for x in np.linspace(-1,1,pupil_n) if x*x+y*y<=1 and (x!=0 or y!=0)]
            nr=len(pupils); traces=[]
            for index,(dx,dy) in enumerate(cfg.dataset.positions_mm):
                set_lens_shift(session,cfg,dx,dy)
                tool=session.system.Tools.OpenBatchRayTrace()
                try:
                    data=tool.CreateNormUnpol(len(field_xy)*nr,session.ZOSAPI.Tools.RayTrace.RaysType.Real,int(session.system.LDE.NumberOfSurfaces)-1)
                    data.ClearData()
                    for x,y in field_xy:
                        for px,py in pupils:
                            if not data.AddRay(primary,float(x/max_field),float(y/max_field),px,py,opd):
                                raise RuntimeError('AddRay rejected input')
                    tool.RunAndWaitForCompletion()
                    data.StartReadingResults()
                    traced=np.full((len(field_xy)*nr,5),np.nan)
                    while True:
                        result=data.ReadNextResult(System.Int32(0),System.Int32(0),System.Int32(0),*[System.Double(0.) for _ in range(11)])
                        if not result[0]: break
                        ray=int(result[1])-1
                        traced[ray]=[float(result[4]),float(result[5]),int(result[2]),int(result[3]),float(result[-1])]
                    if np.isnan(traced[:,2]).any(): raise RuntimeError('Missing batch ray results')
                    traces.append(traced.reshape(len(field_xy),nr,5))
                    print(f'traced pupil={pupil_n}, frame={index}, rays={len(field_xy)*nr}',flush=True)
                finally: tool.Close()
            all_traces=np.array(traces)
            if np.max(np.abs(all_traces[1:,:,:,:2]-all_traces[0:1,:,:,:2]))<1e-10:
                raise RuntimeError('All shifted traces equal center; check batch-tool state caching')
            np.savez_compressed(args.output/f'rays_pupil{pupil_n}.npz',traces=all_traces,field_xy_mm=field_xy,pupil_xy=np.array(pupils))
            center=all_traces[0]
            pixel=cfg.simulation.lr_pixel_size_mm
            for frame,current in enumerate(all_traces):
                for j,obj_xy in enumerate(field_xy):
                    valid=(current[j,:,2]==0)&(current[j,:,3]==0)&(center[j,:,2]==0)&(center[j,:,3]==0)
                    if not valid.any(): raise RuntimeError('No common valid pupil rays')
                    before=center[j,valid,:2]; after=current[j,valid,:2]
                    centroid_delta=(after.mean(0)-before.mean(0))/pixel
                    chief_valid=bool(valid[0])
                    chief_delta=(current[j,0,:2]-center[j,0,:2])/pixel if chief_valid else np.array([np.nan,np.nan])
                    rms_before=float(np.sqrt(np.mean(np.sum((before-before.mean(0))**2,axis=1)))/pixel)
                    rms_after=float(np.sqrt(np.mean(np.sum((after-after.mean(0))**2,axis=1)))/pixel)
                    displacement=(after-before)/pixel
                    rows.append({'pupil_sampling':pupil_n,'frame':frame,'object_xy_mm':obj_xy.tolist(),'center_chief_xy_mm':center[j,0,:2].tolist(),'common_valid_rays':int(valid.sum()),'total_pupil_rays':nr,'chief_valid':chief_valid,'chief_shift_physical_lr_px':chief_delta.tolist(),'chief_shift_png_lr_px':(chief_delta*np.array([1,-1])).tolist(),'centroid_shift_png_lr_px':(centroid_delta*np.array([1,-1])).tolist(),'pupil_displacement_spread_lr_px':float(np.sqrt(np.mean(np.sum((displacement-displacement.mean(0))**2,axis=1)))),'rms_spot_radius_center_lr_px':rms_before,'rms_spot_radius_shifted_lr_px':rms_after,'rms_radius_change_lr_px':rms_after-rms_before})
            set_lens_shift(session,cfg,0.,0.)
        (args.output/'geometry.json').write_text(json.dumps({'metadata':metadata,'rows':rows},indent=2)+'\n')
        print('Saved '+str(args.output/'geometry.json'),flush=True)


if __name__=='__main__': main()
