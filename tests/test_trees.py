import unittest

import numpy as np
import sympy as sp

from symreg.experiment import Experiment
from symreg.models import MODELS
from symreg.trees import D, Field, Grammar, Node, curl, div, log_likelihood, subtree, swap_subtrees

VDP = 'add3(mul(aff(sq(x)), dx), sin(t), x)'


def vdp():
    return Experiment('vdp').eqs[0]


class TestMatrix(unittest.TestCase):
    def test_roundtrip_all_models(self):
        for name in MODELS:
            for extra in (False, True):
                for eq in Experiment(name, extra).eqs:
                    with self.subTest(model=name, extra=extra, target=eq.target):
                        self.assertEqual(str(eq.G.from_matrix(eq.M)), eq.formula)
                        self.assertTrue(np.array_equal(eq.G.to_matrix(eq.G.from_matrix(eq.M)), eq.M))

    def test_layout(self):
        eq = vdp()
        self.assertEqual(eq.G.rows, ['*', 'add3', 'mul', 'aff', 'sq', 'sin'])
        self.assertEqual(eq.G.cols, ['add3', 'mul', 'aff', 'sq', 'sin', 'x', 'dx', 't'])
        self.assertEqual(eq.M[0].tolist(), [1, 0, 0, 0, 0, 0, 0, 0])  # корень -> add3
        self.assertEqual(eq.M.sum(), len(eq.tree))  # рёбра дерева + ребро из корня

    def test_invalid_matrix(self):
        eq = vdp()
        for i, j in [(2, 6), (0, 5)]:  # убрать ребро mul -> dx; добавить второго ребёнка корню
            M = eq.M.copy()
            M[i, j] = 1 - M[i, j]
            with self.subTest(edge=(i, j)), self.assertRaises(ValueError):
                eq.G.from_matrix(M)

    def test_typed_slots(self):
        # МГД: аргументы раскладываются по типам, а не по порядку столбцов
        eq = Experiment('mhd').eqs[0]
        self.assertEqual(str(eq.G.from_matrix(eq.M)), 'div(mul(rho, v))')
        M = eq.M.copy()
        M[2, eq.G.cols.index('rho')], M[2, eq.G.cols.index('B')] = 0, 1  # mul(SV) получает два вектора
        with self.assertRaises(ValueError):
            eq.G.from_matrix(M)


class TestSympy(unittest.TestCase):
    def test_vdp_expression(self):
        eq = vdp()
        mu, A, c_x, a0, a1, omega, x, dx, t = sp.symbols('mu A c_x a0 a1 omega x dx t')
        expected = mu * (a0 + a1 * x ** 2) * dx + A * sp.sin(omega * t) + c_x * x
        self.assertEqual(sp.simplify(eq.expr.subs({s: sp.Symbol(s.name) for s in eq.expr.free_symbols}) - expected), 0)

    def test_evaluate(self):
        eq = vdp()
        x, dx, t = eq.G.symbols
        rng = np.random.default_rng(0)
        data = {x: rng.normal(size=5), dx: rng.normal(size=5), t: rng.normal(size=5)}
        values = {eq.G.params[p]: v for p, v in dict(mu=2, A=3, c_x=-1, a0=1, a1=-1, omega=0.5).items()}
        y = eq.G.evaluate(eq.tree, data, values)
        expected = 2 * (1 - data[x] ** 2) * data[dx] + 3 * np.sin(0.5 * data[t]) - data[x]
        np.testing.assert_allclose(y, expected)

    def test_vector_operators(self):
        v = sp.Matrix([Field('v' + a) for a in 'xyz'])
        rho = Field('rho')
        self.assertEqual(div(curl(v)), 0)  # смешанные производные упорядочены: vx_yz = vx_zy
        self.assertEqual(sp.expand(D(rho * v[0], 'x') - Field('rho_x') * v[0] - rho * Field('vx_x')), 0)


class TestTreeOps(unittest.TestCase):
    def test_zss(self):
        G = Grammar([Node('mul', 'mul', 2), Node('sq', 'sq', 1)], ['x', 'y'])
        a, b, c = G.parse('mul(x, sq(y))'), G.parse('mul(sq(y), x)'), G.parse('mul(x, sq(x))')
        self.assertEqual(G.zss_distance(a, a), 0)
        self.assertEqual(G.zss_distance(a, b), 0)  # порядок аргументов не важен
        self.assertEqual(G.zss_distance(a, c), 1)  # одна замена метки

    def test_subtree_and_swap(self):
        eq = vdp()
        tree = eq.G.parse(VDP)
        self.assertEqual(str(subtree(tree, 2)), 'aff(sq(x))')
        other = eq.G.parse(VDP)
        swap_subtrees(tree, 2, other, 6)  # aff(sq(x)) <-> sin(t)
        self.assertEqual(str(tree), 'add3(mul(sin(t), dx), sin(t), x)')
        self.assertEqual(str(other), 'add3(mul(aff(sq(x)), dx), aff(sq(x)), x)')

    def test_log_likelihood(self):
        M = vdp().M
        P = np.where(M, 0.9, 0.1)
        self.assertAlmostEqual(log_likelihood(M, P), M.size * np.log(0.9))
        self.assertGreater(log_likelihood(M, P), log_likelihood(1 - M, P))


if __name__ == '__main__':
    unittest.main()
