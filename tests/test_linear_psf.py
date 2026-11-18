import unittest
import numpy as np
from super_resolution_imaging.data.linear_psf import unit_flux_kernel,convolve_fixed_scale,detector_average

class LinearPsfTests(unittest.TestCase):
    def test_physical_axis_and_flux(self):
        psf=np.zeros((8,8));psf[5,6]=3
        k=unit_flux_kernel(psf)
        image=np.zeros((21,21));image[10,10]=1
        result=convolve_fixed_scale(image,k)
        self.assertAlmostEqual(float(result.sum()),1.)
        # Physical +x,+y moves right and upward in a PNG.
        self.assertEqual(np.unravel_index(result.argmax(),result.shape),(9,12))

    def test_no_wrapping_or_auto_exposure(self):
        k=np.zeros((3,3));k[1,2]=1
        a=np.zeros((8,8));a[4,7]=1
        self.assertLess(float(np.abs(convolve_fixed_scale(a,k)).max()),1e-14)
        b=np.full((8,8),.25)
        np.testing.assert_allclose(detector_average(b),.25,atol=1e-14)

    def test_additivity_with_realistic_psf(self):
        yy,xx=np.indices((16,16));p=np.exp(-((xx-8)**2+(yy-8)**2)/8.)
        k=unit_flux_kernel(p);rng=np.random.default_rng(5400)
        a=rng.random((32,32))*.3;b=rng.random((32,32))*.2
        render=lambda x:detector_average(convolve_fixed_scale(x,k))
        np.testing.assert_allclose(render(a+b),render(a)+render(b),atol=1e-14)
        np.testing.assert_allclose(render(a*.5),render(a)*.5,atol=1e-14)

if __name__=='__main__':unittest.main()
