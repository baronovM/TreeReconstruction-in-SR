# Деревья формул на DEAP: дерево <-> матрица смежности, поддеревья, SymPy, zss, правдоподобие.
# Матрица M: строки — корень '*' + функции, столбцы — функции + переменные.
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
import zss
from deap import gp

# Примитивы строят SymPy-выражения; w — символы параметров узла.
KINDS = {
    'lin': lambda w: lambda *a: sum(wi * ai for wi, ai in zip(w, a)),
    'aff': lambda w: lambda a: w[0] + w[1] * a,
    'mul': lambda w: lambda a, b: a * b,
    'sq':  lambda w: lambda a: a ** 2,
    'sin': lambda w: lambda a: sp.sin(w[0] * a),
}


@dataclass
class Node:
    name: str
    kind: str
    arity: int
    params: list = field(default_factory=list)  # [(имя, матожидание, ст. отклонение)]


class Grammar:
    def __init__(self, nodes, variables):
        self.nodes, self.variables = nodes, variables
        self.rows = ['*'] + [n.name for n in nodes]
        self.cols = [n.name for n in nodes] + variables
        self.arities = [n.arity for n in nodes]
        self.symbols = sp.symbols(variables)
        self.params = {p: sp.Symbol(p) for n in nodes for p, *_ in n.params}
        self.prior = {self.params[p]: (m, s) for n in nodes for p, m, s in n.params}
        self.pset = gp.PrimitiveSetTyped('F', [float] * len(variables), float)
        self.pset.renameArguments(**{f'ARG{i}': v for i, v in enumerate(variables)})
        for n in nodes:
            w = [self.params[p] for p, *_ in n.params]
            self.pset.addPrimitive(KINDS[n.kind](w), [float] * n.arity, float, name=n.name)

    def parse(self, s):
        return gp.PrimitiveTree.from_string(s, self.pset)

    def to_matrix(self, tree):
        _, edges, labels = gp.graph(tree)
        M = np.zeros((len(self.rows), len(self.cols)), int)
        M[0, self.cols.index(labels[0])] = 1
        for a, b in edges:
            M[self.rows.index(labels[a]), self.cols.index(labels[b])] = 1
        return M

    def from_matrix(self, M):
        nf, out = len(self.nodes), []

        def build(r, depth=0):
            kids = np.flatnonzero(M[r])
            if r and (len(kids) != self.nodes[r - 1].arity or depth > len(self.rows)):
                raise ValueError(f'row {self.rows[r]}: {len(kids)} args')
            if r:
                out.append(self.pset.mapping[self.rows[r]])
            for c in kids:
                if c < nf:
                    build(c + 1, depth + 1)
                else:
                    out.append(self.pset.mapping[self.cols[c]])

        if M[0].sum() != 1:
            raise ValueError('root must have exactly one child')
        build(0)
        return gp.PrimitiveTree(out)

    def to_sympy(self, tree):
        return gp.compile(tree, self.pset)(*self.symbols)

    def evaluate(self, tree, data, values):
        # data — массивы переменных в порядке self.variables; values — {символ параметра: число}
        expr = self.to_sympy(tree)
        ps = sorted(expr.free_symbols - set(self.symbols), key=str)
        f = sp.lambdify([*self.symbols, *ps], expr, 'numpy')
        return np.broadcast_to(f(*data, *[values[p] for p in ps]), np.shape(data[0]))

    def zss_tree(self, tree):
        _, edges, labels = gp.graph(tree)
        kids = {i: [] for i in labels}
        for a, b in edges:
            kids[a].append(b)

        def build(i):  # дети отсортированы: сравнение не зависит от порядка аргументов
            return zss.Node(labels[i], sorted((build(j) for j in kids[i]), key=str))
        return build(0)

    def zss_distance(self, t1, t2):
        return zss.simple_distance(self.zss_tree(t1), self.zss_tree(t2))


def subtree(tree, i):
    return gp.PrimitiveTree(tree[tree.searchSubtree(i)])


def swap_subtrees(t1, i1, t2, i2):
    s1, s2 = t1.searchSubtree(i1), t2.searchSubtree(i2)
    t1[s1], t2[s2] = t2[s2], t1[s1]


def log_likelihood(M, P, eps=1e-9):
    # рёбра — независимые бернуллиевские величины с вероятностями P
    P = np.clip(P, eps, 1 - eps)
    return float((M * np.log(P) + (1 - M) * np.log(1 - P)).sum())
