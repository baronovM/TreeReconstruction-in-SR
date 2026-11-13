import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from symreg.experiment import Experiment, load, noisy
from symreg.models import Fields
from symreg.trees import Field


class TestNoise(unittest.TestCase):
    def test_noisy(self):
        M = Experiment('vdp').eqs[0].M
        U = np.random.default_rng(0).uniform(-1, 1, M.shape)
        np.testing.assert_array_equal(noisy(M, U, 0), M)
        P = noisy(M, U, 0.7)
        self.assertEqual((P.min(), P.max()), (0, 1))


class TestData(unittest.TestCase):
    def test_fields(self):
        f = Fields(np.random.default_rng(0))
        divB = f[Field('Bx_x')] + f[Field('By_y')] + f[Field('Bz_z')]
        self.assertLess(np.abs(divB).max(), 1e-10)
        self.assertLessEqual(np.abs(f[Field('rho')] - 1).max(), 0.5 + 1e-12)
        self.assertAlmostEqual(np.std([f[Field('v' + a)] for a in 'xyz']), 1)  # СКО по всем компонентам

    def test_simulate_shapes(self):
        rng = np.random.default_rng(0)
        for name, shapes in [('vdp', [(1500,)]), ('lv', [(1500,)] * 2), ('mhd', [(12 ** 3,), (3, 12 ** 3), (3, 12 ** 3)])]:
            exp = Experiment(name)
            data, ys = exp.simulate(exp.sample_params(rng), rng)
            with self.subTest(model=name):
                self.assertEqual([np.shape(y) for y in ys], shapes)

    def test_lorenz_derivatives_match_equations(self):
        rng = np.random.default_rng(0)
        exp = Experiment('lorenz')
        values = exp.sample_params(rng)
        data, ys = exp.simulate(values, rng, T=2, n=100)
        x, y, z = (data[Field(axis)] for axis in 'xyz')
        params = {str(p): v for p, v in values.items()}
        sigma, rho, beta = -params['neg_sigma'], params['rho'], -params['neg_beta']
        self.assertEqual(np.shape(ys), (3, 100))
        self.assertTrue(np.isfinite(ys).all())
        np.testing.assert_allclose(ys, [sigma * (y - x), rho * x - x * z - y, x * y - beta * z], atol=1e-10)


class TestEquation(unittest.TestCase):
    def test_structure(self):
        eq = Experiment('vdp', extra=True).eqs[0]
        s = eq.structure(eq.M)
        self.assertTrue(s['sympy'])
        self.assertEqual(s['zss'], 0)
        M = eq.M.copy()
        M[eq.G.rows.index('aff'), eq.G.cols.index('sq')] = 0
        M[eq.G.rows.index('aff'), eq.G.cols.index('sq2')] = 1  # sq -> sq2: та же формула, другое дерево
        M[eq.G.rows.index('sq2'), eq.G.cols.index('x')] = 1
        M[eq.G.rows.index('sq'), eq.G.cols.index('x')] = 0
        s = eq.structure(M)
        self.assertTrue(s['sympy'])
        self.assertEqual(s['zss'], 1)
        M[0] = 0
        self.assertIsNone(eq.structure(M))


class TestRun(unittest.TestCase):
    def test_lv_run_and_save(self):
        exp = Experiment('lv')
        alphas = np.array([0, 1.0])
        acc, zss, invalid = exp.run(repeats=2, trials=2, alphas=alphas)
        self.assertEqual(set(acc), {'dx', 'dy', 'system'})
        for p in acc:
            self.assertEqual(set(acc[p]), {'exact', 'sympy', 'zss=0', 'fit ls+lm', 'fit lm'})
            for c, v in acc[p].items():
                self.assertEqual(v.shape[0], 2)
                self.assertTrue((v[..., 0] == 1).all(), (p, c))  # без шума всё восстанавливается
            self.assertTrue((zss[p][..., 0] == 0).all())
        with tempfile.TemporaryDirectory() as d, mock.patch('symreg.experiment.FIG', Path(d)):
            exp.save(alphas, acc, zss)
            a, methods, z, acc2 = load('lv', 'exact_dict_system')
        np.testing.assert_array_equal(a, alphas)
        np.testing.assert_array_equal(z, zss['system'])
        np.testing.assert_array_equal(acc2['fit ls+lm'], acc['system']['fit ls+lm'])

    def test_seed(self):
        exp = Experiment('lv')
        r1, r2 = (exp.run(1, 2, np.array([0.8]), seed=5)[0]['system']['exact'] for _ in range(2))
        np.testing.assert_array_equal(r1, r2)


if __name__ == '__main__':
    unittest.main()
