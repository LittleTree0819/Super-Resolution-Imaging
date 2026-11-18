"""Export the actual Image Simulation PSF grid without modifying production."""
import json
import argparse
import shutil
from pathlib import Path
import numpy as np
from super_resolution_imaging.config import load_config
from super_resolution_imaging.zos.connection import connect
from super_resolution_imaging.zos.model import set_lens_shift
from super_resolution_imaging.zos.image_simulation import _enum_member, _sample_size

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--sampling',type=int,default=128)
args=parser.parse_args()
out=root/'runs/0276_03_psf_scale_check/psf_grids'
if args.sampling!=128: out=out.parent/('psf_grids_'+str(args.sampling))
out.mkdir(parents=True,exist_ok=True)
cfg=load_config(root/'configs/zemax_quarter_four_frozen.toml')
report={'completed':False,'frames':[]}
with connect(cfg.zemax.install_dir) as session:
    session.system.LoadFile(str(root/'runs/div2k_quarter_four_v1/prepared_model.zmx'),False)
    staged=cfg.zemax.images_dir/'sr_0276_psf_grid.png'
    if staged.exists():raise FileExistsError(staged)
    shutil.copy2(root/'runs/div2k_quarter_four_v1/samples/0276_03/source.png',staged)
    try:
        for i,position in enumerate(cfg.dataset.positions_mm):
            set_lens_shift(session,cfg,*position)
            analysis=session.system.Analyses.New_Analysis(session.ZOSAPI.Analysis.AnalysisIDM.ImageSimulation)
            try:
                s=analysis.GetSettings(); c=cfg.simulation
                s.InputFile=staged.name; s.FieldHeight=c.field_height_mm; s.Field.SetFieldNumber(c.field_number)
                s.PupilSampling=_sample_size(session.ZOSAPI,args.sampling)
                s.ImageSampling=_sample_size(session.ZOSAPI,args.sampling)
                s.PSFXPoints=c.psf_x_points;s.PSFYPoints=c.psf_y_points
                s.Reference=_enum_member(session.ZOSAPI.Analysis.Settings.ReferenceGia,c.reference)
                s.Aberrations=_enum_member(session.ZOSAPI.Analysis.Settings.ExtendedScene.ISAberrationTypes,c.aberrations)
                s.ShowAs=session.ZOSAPI.Analysis.Settings.ExtendedScene.ISShowAsTypes.PSFGrid
                s.XPixels=c.lr_pixels;s.YPixels=c.lr_pixels;s.PixelSize=c.lr_pixel_size_mm
                s.OutputFile=str(out/f'grid_{i:02d}.png')
                analysis.ApplyAndWaitForCompletion()
                r=analysis.GetResults()
                row={'frame':i,'scalar_grids':int(r.NumberOfDataGrids),'rgb_grids':int(r.NumberOfDataGridsRgb),
                     'messages':[str(r.GetMessageAt(k)) for k in range(r.NumberOfMessages)],'grids':[]}
                row['text_exported']=bool(r.GetTextFile(str(out/f'grid_{i:02d}.txt')))
                for k in range(r.NumberOfDataGrids):
                    g=r.GetDataGrid(k)
                    a=np.array(list(g.Values),dtype=float).reshape(int(g.Ny),int(g.Nx))
                    np.save(out/f'grid_{i:02d}_{k:02d}.npy',a)
                    row['grids'].append({'description':str(g.Description),'shape':list(a.shape),
                                         'sum':float(a.sum()),'max':float(a.max()),'finite':bool(np.isfinite(a).all())})
                report['frames'].append(row)
                (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
                print(row,flush=True)
            finally:analysis.Close()
    finally:staged.unlink(missing_ok=True)
report['completed']=True
(out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
