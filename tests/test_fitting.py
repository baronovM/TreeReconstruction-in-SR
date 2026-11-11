import unittest

import numpy as np
import sympy as sp

from symreg.experiment import Experiment
from symreg.fitting import canonical, fit, linear_params, linearize, nonlinear_params
from symreg.trees import Grammar, Node


def names(ps):
    return [p.name for p in ps]


class TestParams(unittest.TestCase):
    def test_vdp(self):
        eq = Experiment('vdp').eqs[0]
        params = list(eq.G.params.values())
        self.assertEqual(names(nonlinear_params(eq.expr, params)), ['omega'])
        # a0 и a1 входят в одночлены вместе с mu, поэтому совместно линейны только mu, A, c_x
        self.assertEqual(names(linear_params(eq.expr, params)), ['mu', 'A', 'c_x'])

    def test_no_nonlinear(self):
        for name in ('lv', 'mhd'):
            for eq in Experiment(name).eqs:
                params = list(eq.G.params.values())
                with self.subTest(model=name, target=eq.target):
                    self.assertEqual(nonlinear_params(eq.expr, params), [])
                    # dm: k_v, k_B стоят в одночленах вместе с k_m, поэтому в МНК идут только k_m, c_p
                    expected = ['k_m', 'c_p'] if eq.target == 'dm' else names(params)
                    self.assertEqual(names(linear_params(eq.expr, params)), expected)

    def test_canonical(self):
        # эквивалентность с точностью до перестановки аргументов и значений линейных коэффициентов
        G = Grammar([Node('add', 'lin', 2, [('p', 1, 0), ('q', 1, 0)]), Node('mul', 'mul', 2),
                     Node('sin', 'sin', 1, [('w', 1, 0)])], ['x', 'y'])
        key = lambda s: canonical(G.to_sympy(G.parse(s)), G.params.values())
        self.assertEqual(key('add(mul(x, y), x)'), key('add(x, mul(y, x))'))
        self.assertEqual(key('add(sin(x), y)'), key('add(y, sin(x))'))
        self.assertNotEqual(key('add(sin(x), y)'), key('add(sin(y), x)'))


class TestFit(unittest.TestCase):
    def check_true_tree(self, name):
        exp = Experiment(name)
        rng = np.random.default_rng(0)
        values = exp.sample_params(rng)
        data, ys = exp.simulate(values, rng)
        for eq, y in zip(exp.eqs, ys):
            for method in ('ls+lm', 'lm'):
                with self.subTest(target=eq.target, method=method):
                    err, found = fit(eq.expr, data, y, eq.G.prior, method)
                    self.assertLess(err, 1e-6)
                    # отдельные параметры неидентифицируемы (ВдП: важны только mu*a0, mu*a1),
                    # поэтому сравниваем коэффициенты при базисных функциях f = sum c_k phi_k
                    variables = eq.expr.free_symbols - set(eq.G.prior)
                    theta = nonlinear_params(eq.expr, eq.G.prior)
                    coefs = linearize(eq.expr, variables, theta)[1]
                    np.testing.assert_allclose([float(c.subs(found)) for c in coefs],
                                               [float(c.subs(values)) for c in coefs], rtol=1e-5)
                    for t in theta:
                        self.assertAlmostEqual(found[t], values[t], places=5)

    def test_vdp(self):
        self.check_true_tree('vdp')

    def test_lv(self):
        self.check_true_tree('lv')

    def test_wrong_tree_fits_badly(self):
        exp = Experiment('vdp')
        rng = np.random.default_rng(0)
        data, (y,) = exp.simulate(exp.sample_params(rng), rng)
        G = exp.eqs[0].G
        wrong = G.to_sympy(G.parse('add3(mul(aff(sq(dx)), x), sin(t), x)'))
        self.assertGreater(fit(wrong, data, y, G.prior)[0], 1e-2)

    def test_vanishing_params(self):
        # dot(cross(v, B), v) = 0 и curl(grad(rho)) = 0: k_m, k_v, k_B исчезают после раскрытия скобок
        exp = Experiment('mhd', extra=True)
        rng = np.random.default_rng(0)
        data, ys = exp.simulate(exp.sample_params(rng), rng)
        G = exp.eqs[1].G
        tree = G.parse('add(divT(sub(outer(mul(dot(cross2(v, B), v), v), v), sq(curl2(grad2(rho))))), grad(rho))')
        for method in ('ls+lm', 'lm'):
            with self.subTest(method=method):
                self.assertGreater(fit(G.to_sympy(tree), data, ys[1], G.prior, method)[0], 0.1)

    def test_unknown_method(self):
        x, p = sp.symbols('x p')
        with self.assertRaises(ValueError):
            fit(p * x, {x: np.ones(3)}, np.ones(3), {p: (1, 0)}, 'nope')


if __name__ == '__main__':
    unittest.main()
