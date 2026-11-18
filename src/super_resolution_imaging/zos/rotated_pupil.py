"""Experimental equivalent coordinate rotation for pupil quadrature studies.

Changes are in-memory only. Geometry and complex-field invariance must be
validated on the current model before this can be used as a reference method.
"""
from __future__ import annotations
import numpy as np
import time
from .model import find_shift_surfaces,set_lens_shift
from .image_simulation import _sample_size

def rotation(angle_deg: float) -> np.ndarray:
    t=np.deg2rad(angle_deg);c=np.cos(t);s=np.sin(t)
    return np.array([[c,-s],[s,c]])

def apply_equivalent_rotation(session,cfg,position,field,wave,angle_deg):
    r=rotation(angle_deg);d=r@np.asarray(position);f=r@np.asarray(field)
    set_lens_shift(session,cfg,*position)
    before,after=find_shift_surfaces(session.system,cfg)
    columns=session.ZOSAPI.Editors.LDE.SurfaceColumn
    for surface in (before,after):surface.GetSurfaceCell(columns.Par6).IntegerValue=0
    before.GetSurfaceCell(columns.Par1).DoubleValue=float(d[0])
    before.GetSurfaceCell(columns.Par2).DoubleValue=float(d[1])
    before.GetSurfaceCell(columns.Par5).DoubleValue=float(angle_deg)
    after.GetSurfaceCell(columns.Par5).DoubleValue=0.
    # T(R*d) R T(-d) = R: image frame co-rotates about its original origin.
    fields=session.system.SystemData.Fields
    fields.GetField(1).X=float(f[0]);fields.GetField(1).Y=float(f[1])
    session.system.SystemData.Wavelengths.GetWavelength(1).Wavelength=float(wave)
    session.system.SystemData.Advanced.HuygensIntegralMethod=session.ZOSAPI.SystemData.HuygensIntegralSettings.Spherical

def native_grid(session,pupil,image_n,delta_um,kind,debug_text=None,strict_finite=True):
    a=session.system.Analyses.New_Analysis_SettingsFirst(session.ZOSAPI.Analysis.AnalysisIDM.HuygensPsf)
    try:
        s=a.GetSettings();s.Field.SetFieldNumber(1);s.Wavelength.SetWavelengthNumber(1)
        s.PupilSampleSize=_sample_size(session.ZOSAPI,pupil);s.ImageSampleSize=_sample_size(session.ZOSAPI,image_n)
        s.ImageDelta=float(delta_um);s.UseCentroid=False;s.UsePolarization=False
        s.ShowAsType=session.ZOSAPI.Analysis.HuygensShowAsTypes.Surface
        s.Type=getattr(session.ZOSAPI.Analysis.Settings.HuygensPsfTypes,kind)
        s.Normalize=False
        outcome=a.ApplyAndWaitForCompletion()
        a.WaitForCompletion()
        if a.IsRunning():raise RuntimeError('Analysis still running')
        if outcome is not None and str(outcome.ErrorCode)!='Success':raise RuntimeError(str(outcome.Text))
        result=a.GetResults()
        if result.NumberOfDataGrids!=1:raise RuntimeError('Missing Huygens grid')
        g=result.GetDataGrid(0);nx=int(g.Nx);ny=int(g.Ny)
        if nx!=image_n or ny!=image_n:raise RuntimeError('Unexpected grid dimensions')
        values=np.fromiter(g.Values,dtype=float,count=nx*ny).reshape(nx,ny)
        if debug_text is not None:result.GetTextFile(str(debug_text))
        # Current Huygens export has an unused first row/column: Linear fills
        # these with zero, while complex modes may expose uninitialized cells.
        # Exclude them by coordinate, never repair any invalid interior value.
        values=values[1:,1:].copy()
        invalid=np.argwhere(~np.isfinite(values))
        if strict_finite and len(invalid):raise RuntimeError(f'Nonfinite interior {kind} field; see exported text')
        return values,{'dx_um':float(g.Dx),'dy_um':float(g.Dy),'minx_um':float(g.MinX+g.Dx),'miny_um':float(g.MinY+g.Dy),'array_axes':'x,y',
                       'excluded_unused_first_row_column':True,'raw_image_n':image_n,
                       'nonfinite_indices':invalid[:32].tolist(),
                       'actual_type':str(s.Type),'actual_normalize':bool(s.Normalize),'show_as':str(s.ShowAsType),
                       'messages':[{'code':str(result.GetMessageAt(k).ErrorCode),'text':str(result.GetMessageAt(k).Text)} for k in range(result.NumberOfMessages)]}
    finally:a.Close()

def trace_probe(session,physical_field,pupils,normalization_radius=56.):
    import System
    tool=session.system.Tools.OpenBatchRayTrace()
    try:
        data=tool.CreateNormUnpol(len(pupils),session.ZOSAPI.Tools.RayTrace.RaysType.Real,int(session.system.LDE.NumberOfSurfaces)-1)
        data.ClearData();opd=System.Enum.Parse(session.ZOSAPI.Tools.RayTrace.OPDMode,'None')
        for px,py in pupils:
            if not data.AddRay(1,float(physical_field[0]/normalization_radius),float(physical_field[1]/normalization_radius),float(px),float(py),opd):
                raise RuntimeError('Ray rejected')
        tool.RunAndWaitForCompletion();data.StartReadingResults()
        out=np.full((len(pupils),10),np.nan)
        while True:
            v=data.ReadNextResult(System.Int32(0),System.Int32(0),System.Int32(0),*[System.Double(0.) for _ in range(11)])
            if not v[0]:break
            out[int(v[1])-1]=[int(v[2]),int(v[3]),*[float(t) for t in v[4:10]],float(v[-2]),float(v[-1])]
        if not np.isfinite(out).all():raise RuntimeError('Incomplete ray probe')
        return out
    finally:tool.Close()

def native_complex_roi(session,pupil,image_n,delta_um,chief_xy_mm,progress=None):
    """Read and verify L,Re,Im on a declared physical detector ROI only.

    Nonfinite values anywhere in the required ROI and a failed complex closure
    are hard failures. No gain or phase alignment is fitted to the result.
    Reuse of one analysis permits the engine to retain geometry between types.
    """
    a=session.system.Analyses.New_Analysis_SettingsFirst(session.ZOSAPI.Analysis.AnalysisIDM.HuygensPsf)
    arrays={};timings={};grid=None
    try:
        s=a.GetSettings();s.Field.SetFieldNumber(1);s.Wavelength.SetWavelengthNumber(1)
        s.PupilSampleSize=_sample_size(session.ZOSAPI,pupil);s.ImageSampleSize=_sample_size(session.ZOSAPI,image_n)
        s.ImageDelta=float(delta_um);s.UseCentroid=False;s.UsePolarization=False
        s.ShowAsType=session.ZOSAPI.Analysis.HuygensShowAsTypes.Surface
        for kind in ('Linear','Real','Imaginary'):
            if progress is not None:progress(kind)
            s.Type=getattr(session.ZOSAPI.Analysis.Settings.HuygensPsfTypes,kind);s.Normalize=False
            started=time.monotonic();a.ApplyAndWaitForCompletion();a.WaitForCompletion()
            if a.IsRunning():raise RuntimeError('Analysis not complete')
            result=a.GetResults()
            if result.NumberOfDataGrids!=1:raise RuntimeError('Missing complex output')
            g=result.GetDataGrid(0);nx=int(g.Nx);ny=int(g.Ny)
            full=np.fromiter(g.Values,dtype=float,count=nx*ny).reshape(nx,ny)
            xs=float(g.MinX)+np.arange(nx)*float(g.Dx);ys=float(g.MinY)+np.arange(ny)*float(g.Dy)
            mx=(np.abs(xs+chief_xy_mm[0]*1000)<=153.6+1e-8)&(np.arange(nx)>0)
            my=(np.abs(ys+chief_xy_mm[1]*1000)<=153.6+1e-8)&(np.arange(ny)>0)
            current={'x_um':xs[mx].tolist(),'y_um':ys[my].tolist(),'array_axes':'x,y'}
            if grid is not None and current!=grid:raise RuntimeError('Complex grids differ')
            grid=current;values=full[np.ix_(mx,my)]
            if values.size<25 or not np.isfinite(values).all():raise RuntimeError('Invalid required ROI values')
            arrays[kind]=values.copy();timings[kind]=time.monotonic()-started
        intensity=arrays['Linear'];field=arrays['Real']+1j*arrays['Imaginary']
        if intensity.sum()<=0 or np.any(intensity<0):raise RuntimeError('Invalid intensity')
        closure=float(np.abs(np.abs(field)**2-intensity).sum()/intensity.sum())
        if closure>1e-8:raise RuntimeError(f'Complex closure failed: {closure}')
        return field,intensity,{'grid':grid,'closure_relative_l1':closure,'timings_s':timings,
                                'no_gain_or_phase_fitting':True,'no_output_normalization':True}
    finally:a.Close()
