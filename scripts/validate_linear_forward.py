"""Demonstrate fixed-scale rendering with real PSFs; explicitly NOT release-ready.

This control uses a field-invariant PSF and the cached 0.5876-um chief-ray
geometry. It cannot establish full spatial/chromatic/illumination fidelity.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from super_resolution_imaging.data.linear_psf import unit_flux_kernel,convolve_fixed_scale,detector_average

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/photometric_forward_audit_v1/linear_forward'
OUT.mkdir(parents=True,exist_ok=True)
raw=OUT.parent/'raw_psf';geometry=ROOT/'runs/eight_frame_quarter_geometry'
geo=json.loads((geometry/'geometry.json').read_text(encoding='utf-8'))
if geo['metadata']['source_model']['sha256']!='6e24d7040d91ded19fc901231be4fdaf051613f85cda9cb49b576a1411c93d29':
    raise RuntimeError('Cached geometry model mismatch')
traces=np.load(geometry/'rays_pupil33.npz')['traces']
xy=np.asarray(geo['metadata']['object_grid_xy_mm'])
design=np.c_[xy,np.ones(len(xy))]
source=np.asarray(Image.open(ROOT/'runs/div2k_quarter_four_v1/samples/0276_03/source.png').convert('L'),dtype=float)/255

def warp(a,m,size=384):
    yy,xx=np.indices((size,size));physical=np.stack(((xx+.5-size/2)*.0012,(size/2-yy-.5)*.0012),axis=-1)
    object_xy=(physical-m[2])@np.linalg.inv(m[:2])
    u=object_xy[...,0]/(5.7/a.shape[1])+a.shape[1]/2-.5
    v=-object_xy[...,1]/(5.7/a.shape[0])+a.shape[0]/2-.5
    x0=np.floor(u).astype(int);y0=np.floor(v).astype(int);fx=u-x0;fy=v-y0
    result=np.zeros((size,size))
    for dx,dy,w in [(0,0,(1-fx)*(1-fy)),(1,0,fx*(1-fy)),(0,1,(1-fx)*fy),(1,1,fx*fy)]:
        x=x0+dx;y=y0+dy;valid=(x>=0)&(x<a.shape[1])&(y>=0)&(y<a.shape[0])
        result[valid]+=a[y[valid],x[valid]]*w[valid]
    return result

report={'completed':False,'production_ready':False,'fixed_gain':1.,'scene_normalization':False,
        'throughput_model':'Unit-flux PSFs: constant throughput assumed, not measured.',
        'limitations':['Field-invariant center PSF only','Chief-ray geometry at original 0.5876 um, not RGB chromatic warp',
                       'PSF convergence and support must pass separately','No relative-illumination weighting; not a replacement dataset'],
        'positions':[]}
preview=Image.new('RGB',(768,590),'white');draw=ImageDraw.Draw(preview)
for col,(name,index) in enumerate([('a',0),('b0',4),('b1',5)]):
    if not np.all(traces[index,:,0,2:4]==0):raise RuntimeError('Invalid/vignetted chief rays')
    chief=traces[index,:,0,:2];m=np.linalg.lstsq(design,chief,rcond=None)[0]
    residual=chief-design@m
    kernels=[]
    for wave,weight in ((.606,.299),(.535,.587),(.465,.114)):
        path=raw/f'{name}_center_p512_i128_w{wave:.3f}.npz'
        # ZOS-API Values are indexed x,y; the image operator expects y,x.
        kernels.append(unit_flux_kernel(np.load(path)['psf'].T)*weight)
    k=sum(kernels)
    def render(a):
        wide=convolve_fixed_scale(warp(a,m),k)
        return detector_average(wide[64:320,64:320])
    image=render(source);half=render(source*.5);gray=render(np.full_like(source,.25))
    combined=render(source*.5+.25)
    target=warp(source,m)[64:320,64:320]
    np.savez_compressed(OUT/f'{name}_float.npz',image=image,half=half,gray=gray,combined=combined,
                        ideal_hr=target,kernel=k,affine=m)
    row={'position':name,'mean':float(image.mean()),'range':[float(image.min()),float(image.max())],
         'additivity_max_error':float(np.abs(combined-half-gray).max()),
         'half_scale_max_error':float(np.abs(half-image*.5).max()),
         'chief_affine_residual_max_lr_px':float(np.linalg.norm(residual,axis=1).max()/.0024),
         'gray_mean':float(gray.mean())}
    report['positions'].append(row)
    draw.text((col*256+8,8),f'{name}: fixed gain=1, mean={image.mean():.4f}',fill='black')
    preview.paste(Image.fromarray(np.uint8(np.clip(image,0,1)*255)).convert('RGB').resize((256,256)),(col*256,32))
    draw.text((col*256+8,302),'Ideal HR geometry control',fill='black')
    preview.paste(Image.fromarray(np.uint8(np.clip(target,0,1)*255)).convert('RGB'),(col*256,328))
report['completed']=True
(OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
preview.save(OUT/'preview.png');print(json.dumps(report,indent=2))
