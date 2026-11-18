import unittest
import numpy as np
from super_resolution_imaging.data.complex_quadrature import CoherentMean,dyadic_angles,intensity_error

class CoherentQuadratureTests(unittest.TestCase):
    def test_destructive_interference_not_intensity_average(self):
        mean=CoherentMean();mean.add(np.array([1+0j]));mean.add(np.array([-1+0j]))
        self.assertEqual(float(mean.intensity[0]),0.)

    def test_known_phasor_sum_and_no_input_mutation(self):
        a=np.array([1+0j]);mean=CoherentMean();mean.add(a,2);mean.add(np.array([1j]),1)
        np.testing.assert_allclose(mean.field,[(2+1j)/3]);np.testing.assert_allclose(a,[1])
        self.assertAlmostEqual(float(mean.intensity[0]),5/9)

    def test_nested_unique_directions_and_scale_sensitive_metrics(self):
        self.assertEqual(dyadic_angles(4),[0.,45.,22.5,67.5])
        self.assertEqual(dyadic_angles(8)[:4],dyadic_angles(4))
        self.assertEqual(len(set(dyadic_angles(16))),16)
        errors=intensity_error(np.ones((2,2))*1.1,np.ones((2,2)))
        self.assertAlmostEqual(errors['relative_l1_intensity'],.1)
        self.assertAlmostEqual(errors['shape_tv'],0.)

if __name__=='__main__':unittest.main()
