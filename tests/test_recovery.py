import unittest

import numpy as np

from symreg.experiment import Experiment, noisy
from symreg.models import MODELS
from symreg.recovery import METHODS, restore
from symreg.trees import Grammar, Node


def equations():
    for name in MODELS:
        for extra in (False, True):
            for eq in Experiment(name, extra).eqs:
                yield f'{name} extra={extra} {eq.target}', eq


class TestRestore(unittest.TestCase):
    def test_noiseless_is_exact(self):
        for tag, eq in equations():
            for method in METHODS:
                with self.subTest(eq=tag, method=method):
                    self.assertTrue(np.array_equal(restore(method, eq.M.astype(float), eq.G.ins, eq.G.types), eq.M))

    def test_small_noise_is_exact(self):
        rng = np.random.default_rng(0)
        for tag, eq in equations():
            P = noisy(eq.M, rng.uniform(-1, 1, eq.M.shape), 0.3)
            for method in ('dfs', 'bfs', 'prim'):
                with self.subTest(eq=tag, method=method):
                    self.assertTrue(np.array_equal(restore(method, P, eq.G.ins, eq.G.types), eq.M))

    def test_output_is_typed_adjacency(self):
        # 0/1-матрица той же формы; ребро ведёт только в столбец типа, который есть среди слотов строки
        rng = np.random.default_rng(1)
        for tag, eq in equations():
            P = noisy(eq.M, rng.uniform(-1, 1, eq.M.shape), 1.0)
            for method in METHODS:
                A = restore(method, P, eq.G.ins, eq.G.types)
                with self.subTest(eq=tag, method=method):
                    self.assertEqual(A.shape, P.shape)
                    self.assertTrue(set(np.unique(A)) <= {0, 1})
                    for i, j in np.argwhere(A):
                        self.assertIn(eq.G.types[j], eq.G.ins[i])

    def test_unknown_method(self):
        eq = next(equations())[1]
        with self.assertRaises(ValueError):
            restore('nope', eq.M, eq.G.ins, eq.G.types)

    @unittest.expectedFailure
    def test_fallback_picks_distinct_variables(self):
        # известный баг: все вероятности строки mul < eps -> оба слота берут одну и ту же лучшую переменную
        G = Grammar([Node('mul', 'mul', 2)], ['x', 'y'])
        P = np.array([[1., 0, 0], [0, 0, 0]])
        for method in ('dfs', 'bfs'):
            with self.subTest(method=method):
                self.assertEqual(restore(method, P, G.ins, G.types)[1].sum(), 2)


if __name__ == '__main__':
    unittest.main()
