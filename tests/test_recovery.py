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

    def test_fallback_picks_distinct_variables(self):
        G = Grammar([Node('mul', 'mul', 2)], ['x', 'y', 'z'])
        for best in (2e-5, 0.8):
            P = np.array([[1., 0, 0, 0], [0, 0, best, 1e-5]])
            for method in METHODS:
                if method.endswith('prim'):
                    continue
                with self.subTest(method=method, best=best):
                    A = restore(method, P, G.ins, G.types)
                    self.assertEqual(str(G.from_matrix(A)), 'mul(y, z)')

    def test_fallback_can_reuse_variable_for_another_parent(self):
        G = Grammar([Node('mul', 'mul', 2), Node('sq', 'sq', 1)], ['x'])
        P = np.array([[1., 0, 0], [0, 1, 0], [0, 0, 0]])
        for method in ('dfs', 'bfs'):
            with self.subTest(method=method):
                A = restore(method, P, G.ins, G.types)
                self.assertEqual(str(G.from_matrix(A)), 'mul(sq(x), x)')

    def test_fallback_leaves_slot_empty_when_no_leaf_is_available(self):
        G = Grammar([Node('mul', 'mul', 2)], ['x'])
        P = np.array([[1., 0], [0, 0]])
        for method in ('dfs', 'bfs'):
            with self.subTest(method=method):
                A = restore(method, P, G.ins, G.types)
                np.testing.assert_array_equal(A, [[1, 0], [0, 1]])
                with self.assertRaises(ValueError):
                    G.from_matrix(A)


if __name__ == '__main__':
    unittest.main()
