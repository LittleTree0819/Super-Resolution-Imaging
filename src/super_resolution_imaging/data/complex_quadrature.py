"""Sequential coherent quadrature: bounded memory, no phase/gain fitting."""
from __future__ import annotations
import numpy as np

class CoherentMean:
    def __init__(self):
        self.total=None;self.correction=None;self.weight=0.;self.count=0

    def add(self,field,weight=1.):
        a=np.asarray(field,dtype=np.complex128)
        if not np.isfinite(a).all() or not np.isfinite(weight) or weight<=0:
            raise ValueError('Finite fields and positive weights required')
        if self.total is None:
            self.total=np.zeros_like(a);self.correction=np.zeros_like(a)
        if a.shape!=self.total.shape:raise ValueError('Quadrature grids differ')
        y=a*weight-self.correction;t=self.total+y
        self.correction=(t-self.total)-y;self.total=t
        self.weight+=weight;self.count+=1

    @property
    def field(self):
        if not self.count:raise ValueError('No quadrature blocks')
        return self.total/self.weight

    @property
    def intensity(self):return np.abs(self.field)**2

def intensity_error(a,b):
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.sum(),b.sum())<=0:
        raise ValueError('Invalid intensity comparison')
    return {'relative_l1_intensity':float(np.abs(a-b).sum()/b.sum()),
            'shape_tv':float(np.abs(a/a.sum()-b/b.sum()).sum()/2)}

def dyadic_angles(count):
    """Nested evenly spaced directions in [0,90); 90 degrees is duplicate."""
    if count<1 or count&(count-1):raise ValueError('Direction count must be a power of two')
    bits=count.bit_length()-1
    return [90*int(f'{i:0{bits}b}'[::-1],2)/count for i in range(count)] if count>1 else [0.]
