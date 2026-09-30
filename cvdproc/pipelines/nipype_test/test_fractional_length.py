"""Small exactness tests for the fractional-length disconnection operator."""
import unittest

import numpy as np
from scipy import sparse

from cvdproc.pipelines.multi.disconnection.disconnection_nipype import _fractional_length_weights


class FractionalLengthOperatorTest(unittest.TestCase):
    def setUp(self):
        self.a = sparse.csr_matrix(np.asarray([
            [1.0, 2.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
        ], dtype=np.float64))
        self.p = sparse.csr_matrix(np.asarray([
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ], dtype=np.float64))
        lengths = np.asarray(self.a.sum(axis=1)).ravel()
        inverse_lengths = np.divide(1.0, lengths, out=np.zeros_like(lengths), where=lengths > 0)
        denominator = np.asarray(self.p.T @ np.ones(self.p.shape[0])).ravel()
        inverse_denominator = np.divide(1.0, denominator, out=np.zeros_like(denominator), where=denominator > 0)
        self.w = sparse.diags(inverse_denominator) @ self.p.T @ sparse.diags(inverse_lengths) @ self.a

    def forward(self, lesion):
        affected = np.asarray(self.a @ lesion).ravel()
        lengths = np.asarray(self.a.sum(axis=1)).ravel()
        weights = _fractional_length_weights(affected, lengths)
        numerator = np.asarray(self.p.T @ weights).ravel()
        denominator = np.asarray(self.p.T @ np.ones(self.p.shape[0])).ravel()
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)

    def test_small_example_and_zero_length_streamline(self):
        lesion = np.asarray([1.0, 0.5, 0.0, 0.0])
        np.testing.assert_allclose(_fractional_length_weights(self.a @ lesion, self.a.sum(axis=1)),
                                   [2.0 / 3.0, 0.125, 0.0], rtol=0, atol=1e-15)
        np.testing.assert_allclose(self.forward(lesion), self.w @ lesion, rtol=0, atol=1e-15)
        self.assertEqual(self.forward(lesion)[-1], 0)

    def test_linearity(self):
        first = np.asarray([0.1, 0.4, 0.0, 0.2])
        second = np.asarray([0.7, 0.0, 0.3, 0.1])
        alpha, beta = 0.25, 0.6
        np.testing.assert_allclose(self.forward(alpha * first + beta * second),
                                   alpha * self.forward(first) + beta * self.forward(second),
                                   rtol=0, atol=2e-16)

    def test_additivity(self):
        first = np.asarray([0.1, 0.2, 0.0, 0.0])
        second = np.asarray([0.3, 0.0, 0.1, 0.2])
        np.testing.assert_allclose(self.forward(first + second),
                                   self.forward(first) + self.forward(second), rtol=0, atol=2e-16)

    def test_forward_adjoint_consistency(self):
        lesion = np.asarray([0.2, 0.6, 0.1, 0.3])
        beta_d = np.asarray([0.4, -0.2, 0.8, 0.3])
        beta_l = self.w.T @ beta_d
        self.assertAlmostEqual(float(beta_d @ self.forward(lesion)), float(beta_l @ lesion), places=14)


if __name__ == "__main__":
    unittest.main()
