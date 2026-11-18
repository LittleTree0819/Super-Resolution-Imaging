"""Physical-grid PSF comparisons; arrays follow ZOS-API Values[x_index,y_index]."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PsfGrid:
    values: np.ndarray
    dx_um: float
    dy_um: float
    minx_um: float
    miny_um: float

    @property
    def integral(self) -> float:
        return float(np.sum(self.values,dtype=np.float64)*self.dx_um*self.dy_um)

    def sample(self,x_um: np.ndarray,y_um: np.ndarray) -> np.ndarray:
        """Bilinear intensity interpolation, with no alignment or gain fitting."""
        u=(np.asarray(x_um)-self.minx_um)/self.dx_um
        v=(np.asarray(y_um)-self.miny_um)/self.dy_um
        u,v=np.broadcast_arrays(u,v)
        if np.any(u < -1e-8) or np.any(v < -1e-8) or np.any(u>self.values.shape[0]-1+1e-8) or np.any(v>self.values.shape[1]-1+1e-8):
            raise ValueError('Requested coordinates outside PSF grid')
        u=np.clip(u,0,self.values.shape[0]-1);v=np.clip(v,0,self.values.shape[1]-1)
        i=np.floor(u).astype(int);j=np.floor(v).astype(int)
        i1=np.minimum(i+1,self.values.shape[0]-1);j1=np.minimum(j+1,self.values.shape[1]-1)
        fx=u-i;fy=v-j
        return self.values[i,j]*(1-fx)*(1-fy)+self.values[i1,j]*fx*(1-fy)+self.values[i,j1]*(1-fx)*fy+self.values[i1,j1]*fx*fy


def compare_grid(reference: PsfGrid,candidate: PsfGrid) -> dict[str,float]:
    """Compare on the reference physical coordinates without oracle registration.

    Total variation here refers to the sampled common region. If the reference
    is sparse, these are point-intensity checks, NOT energy convergence gates.
    """
    x=reference.minx_um+np.arange(reference.values.shape[0])*reference.dx_um
    y=reference.miny_um+np.arange(reference.values.shape[1])*reference.dy_um
    a=np.asarray(reference.values,dtype=float);b=candidate.sample(x[:,None],y[None,:])
    if a.sum()<=0 or b.sum()<=0:raise ValueError('Nonpositive PSF')
    return {'relative_l1_intensity':float(np.abs(a-b).sum()/np.abs(a).sum()),
            'normalized_shape_tv':float(np.abs(a/a.sum()-b/b.sum()).sum()/2),
            'max_error_over_reference_peak':float(np.abs(a-b).max()/a.max()),
            'sampled_sum_relative_change':float(abs(b.sum()/a.sum()-1))}
