import unittest
import numpy as np
from super_resolution_imaging.zos.psf_metrics import PsfGrid,compare_grid

class PsfMetricTests(unittest.TestCase):
    def test_integral_accounts_for_area(self):
        def gaussian(n,d):
            x=(np.arange(n)-n//2)*d
            return PsfGrid(np.exp(-(x[:,None]**2+x[None,:]**2)/8),d,d,-n//2*d,-n//2*d)
        a=gaussian(64,.5);b=gaussian(128,.25)
        self.assertAlmostEqual(a.integral,8*np.pi,places=10)
        self.assertAlmostEqual(a.integral,b.integral,places=10)

    def test_coordinates_and_axes_are_not_swapped(self):
        x=np.arange(8)*.5-2;y=np.arange(12)*.25-1.5
        g=PsfGrid(2*x[:,None]+3*y[None,:]+10,.5,.25,-2,-1.5)
        self.assertAlmostEqual(float(g.sample(np.array(-.25),np.array(.125))),9.875)
        with self.assertRaises(ValueError):g.sample(np.array(20),np.array(0))

    def test_interpolation_does_not_hide_gain_error(self):
        a=PsfGrid(np.ones((8,8)),1.,1.,-4.,-4.)
        b=PsfGrid(np.full((16,16),1.1),.5,.5,-4.,-4.)
        c=compare_grid(a,b)
        self.assertAlmostEqual(c['relative_l1_intensity'],.1)
        self.assertAlmostEqual(c['normalized_shape_tv'],0.)

if __name__=='__main__':unittest.main()
