"""Report only measured convergence, distinguishing sparse probes from energy."""
import json
from pathlib import Path
import numpy as np
from super_resolution_imaging.zos.psf_metrics import PsfGrid,compare_grid

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/psf_reliability_v2'
def load(key):
    m=json.loads((OUT/(key+'.json')).read_text(encoding='utf-8'));a=np.load(OUT/(key+'.npz'))['psf']
    return m,PsfGrid(a,m['delta_um'],m['delta_um'],m['minx_um'],m['miny_um'])
def key(p,n,d,mode='Spherical'):
    return f'{mode}_p{p}_i{n}_d{d:g}_a_x0_y0_w0.535'
summary={'production_ready':False,'dataset_modified':False,'array_axes':'ZOS-API x,y; transpose explicitly for image y,x',
         'sparse_warning':'Sparse node metrics are point-intensity checks, never a PSF energy integral or a render kernel.',
         'method_tests':[],'pupil_pairs':[],'node_pairs':[]}
for n in (128,256):
    _,a=load(key(512,n,1.2,'Auto'))
    for method in ('Planar','Spherical'):
        _,b=load(key(512,n,1.2,method))
        summary['method_tests'].append({'image_n':n,'comparison':'Auto vs '+method,'max_absolute_difference':float(np.abs(a.values-b.values).max())})
for mode in ('Planar','Spherical'):
    _,a=load(key(512,128,1.2,mode));_,b=load(key(512,256,1.2,mode))
    summary.setdefault('common_window_tests',[]).append({'method':mode,**compare_grid(a,b)})
for p,q in ((256,512),(512,1024),(1024,2048)):
    _,a=load(key(p,128,1.2));_,b=load(key(q,128,1.2))
    summary['pupil_pairs'].append({'pupil_pair':[p,q],**compare_grid(a,b)})
for d in (4.8,38.4):
    for p,q in ((2048,4096),(4096,8192),(8192,16384)):
        if not (OUT/(key(q,32,d)+'.json')).exists():continue
        _,a=load(key(p,32,d));_,b=load(key(q,32,d))
        summary['node_pairs'].append({'pupil_pair':[p,q],'delta_um':d,'scope':'all probe points',**compare_grid(a,b)})
        if d==38.4:
            coords=a.minx_um+np.arange(32)*d;mask=np.abs(coords)<=153.6+1e-6
            selected=PsfGrid(a.values[np.ix_(mask,mask)],d,d,float(coords[mask][0]),float(coords[mask][0]))
            summary['node_pairs'].append({'pupil_pair':[p,q],'delta_um':d,'scope':'center field detector +/-153.6 um',**compare_grid(selected,b)})
fft=[]
for p in (4096,8192):
    folder=OUT/'fft_reference/output_2048'
    m=json.loads((folder/f'a_center_p{p}.json').read_text(encoding='utf-8'))
    a=np.load(folder/f'a_center_p{p}.npz')['psf']
    fft.append(PsfGrid(a,m['dx_um'],m['dy_um'],m['minx_um'],m['miny_um']))
coords=np.arange(-153.6,153.6,.6)
a=PsfGrid(fft[0].sample(coords[:,None],coords[None,:]),.6,.6,-153.6,-153.6)
summary['fft_core_convergence']=compare_grid(a,fft[1])
for d in (4.8,38.4):
    _,a=load(key(8192,32,d))
    if d==38.4:
        coords=a.minx_um+np.arange(32)*d;mask=np.abs(coords)<=153.6+1e-6
        a=PsfGrid(a.values[np.ix_(mask,mask)],d,d,float(coords[mask][0]),float(coords[mask][0]))
    summary.setdefault('fft_vs_huygens',[]).append({'huygens_pupil':8192,'probe_delta_um':d,**compare_grid(a,fft[1])})
# Cached ray footprints are a sizing aid, not a diffraction energy measurement.
geo=ROOT/'runs/eight_frame_quarter_geometry'
data=np.load(geo/'rays_pupil33.npz')['traces'];chief=data[:,:25,0,:2]
center=data[0,12];valid=(center[:,2]==0)&(center[:,3]==0)
r=np.linalg.norm(center[valid,:2]-center[0,:2],axis=1)
summary['geometric_sizing']={'wavelength_um':.5876,'valid_rays':int(valid.sum()),
                            'radius_mm_percentiles':dict(zip(('50','90','95','99','100'),map(float,np.percentile(r,[50,90,95,99,100])))),
                            'required_kernel_half_width_mm_from_source_and_detector':float(np.abs(chief).max()+.3072/2)}
summary['status']='Measured convergence pending interpretation; production remains on hold.'
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
