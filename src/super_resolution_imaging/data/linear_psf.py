"""Experimental, fixed-scale floating PSF operator; not a release generator.

Kernels are calibrated independently of scene content. This module performs no
peak/mean normalization, clipping or automatic exposure on rendered images.
It does not infer absolute radiometry or relative illumination from PSF values.
"""
from __future__ import annotations
import numpy as np


def unit_flux_kernel(psf: np.ndarray) -> np.ndarray:
    """Convert an even physical x/y grid to an odd, PNG-oriented shape kernel.

    Input sample (n/2,n/2) denotes zero displacement, y increases upwards.
    Unit sum is an explicit constant-throughput modelling assumption, not a
    measurement of the lens transmission. It is applied once to the PSF.
    """
    a=np.asarray(psf,dtype=np.float64)
    if a.ndim!=2 or a.shape[0]!=a.shape[1] or a.shape[0]%2:
        raise ValueError('Expected an even square physical PSF grid')
    if not np.isfinite(a).all() or np.any(a<0) or a.sum()<=0:
        raise ValueError('PSF must be finite, nonnegative, and have positive flux')
    padded=np.pad(a,((0,1),(0,1)))
    return padded[::-1,:]/a.sum()


def convolve_fixed_scale(image: np.ndarray,kernel: np.ndarray) -> np.ndarray:
    """Zero-padded linear convolution, with fixed odd-kernel center and crop."""
    a=np.asarray(image,dtype=np.float64);k=np.asarray(kernel,dtype=np.float64)
    if a.ndim!=2 or k.ndim!=2 or any(n%2==0 for n in k.shape):
        raise ValueError('A 2-D image and an odd 2-D kernel are required')
    if not np.isfinite(a).all() or not np.isfinite(k).all():
        raise ValueError('Nonfinite input')
    full=tuple(x+y-1 for x,y in zip(a.shape,k.shape))
    result=np.fft.irfft2(np.fft.rfft2(a,s=full)*np.fft.rfft2(k,s=full),s=full)
    cy,cx=(n//2 for n in k.shape)
    return result[cy:cy+a.shape[0],cx:cx+a.shape[1]]


def detector_average(image: np.ndarray,factor: int=2) -> np.ndarray:
    """Fixed area averaging, without scene-dependent gain or clipping."""
    a=np.asarray(image,dtype=np.float64)
    if a.ndim!=2 or factor<1 or a.shape[0]%factor or a.shape[1]%factor:
        raise ValueError('Detector dimensions must divide the image')
    return a.reshape(a.shape[0]//factor,factor,a.shape[1]//factor,factor).mean((1,3))
