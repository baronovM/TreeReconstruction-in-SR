# Деревья формул на DEAP: дерево <-> матрица смежности, поддеревья, SymPy, zss, правдоподобие.
# Матрица M: строки — корень '*' + функции, столбцы — функции + переменные.
# Типы DEAP: скаляр — float, вектор — Vec, тензор 2-го ранга — Ten; сигнатура узла 'SV>V'.
from dataclasses import dataclass, field
from functools import reduce
from operator import add

import numpy as np
import sympy as sp
import zss
from deap import gp


class Vec: pass
class Ten: pass
TYPES = {'S': float, 'V': Vec, 'T': Ten}
AXES = 'xyz'
E = sp.symbols('e_x e_y e_z')  # векторное выражение f -> скаляр sum_i e_i f_i


class Field(sp.Symbol):
    pass  # значение поля или его производной в точке: rho, vx, vx_y, vx_yz


def D(e, a):  # d/da по правилу цепочки; смешанные производные упорядочены
    def d(s):
        base, _, axes = s.name.partition('_')
        return Field(f'{base}_{"".join(sorted(axes + a))}')
    return sp.Add(*[sp.diff(e, s) * d(s) for s in e.atoms(Field)])


def div(a):  # вектор -> скаляр, тензор -> вектор (свёртка по первому индексу)
    r = sum((a[i, :].applyfunc(lambda e: D(e, x)) for i, x in enumerate(AXES)), sp.zeros(1, a.cols))
    return r[0] if a.cols == 1 else r.T


def curl(a):
    d = lambda i, x: D(a[i], x)
    return sp.Matrix([d(2, 'y') - d(1, 'z'), d(0, 'z') - d(2, 'x'), d(1, 'x') - d(0, 'y')])


op = lambda f: lambda w: lambda *a: sp.Mul(*w) * f(*a)  # параметры узла — общий множитель
# Примитивы строят SymPy-выражения; w — символы параметров узла.
KINDS = {
    'lin': lambda w: lambda *a: reduce(add, (wi * ai for wi, ai in zip(w, a))),
    'aff': lambda w: lambda a: w[0] + w[1] * a,
    'sin': lambda w: lambda a: sp.sin(w[0] * a),
    'mul': op(lambda a, b: a * b),
    'sub': op(lambda a, b: a - b),
    'sq': op(lambda a: a * a.T if isinstance(a, sp.MatrixBase) else a ** 2),  # вектор: a⊗a
    'dot': op(lambda a, b: a.dot(b)),
    'cross': op(lambda a, b: a.cross(b)),
    'outer': op(lambda a, b: a * b.T),
    'grad': op(lambda a: sp.Matrix([D(a, x) for x in AXES])),
    'div': op(div),
    'curl': op(curl),
}


@dataclass
class Node:
    name: str
    kind: str
    arity: int
    params: list = field(default_factory=list)  # [(имя, матожидание, ст. отклонение)]
    sig: str = None  # типы аргументов и результата, 'SV>V'; по умолчанию всё скаляры

    @property
    def ins(self):
        return [TYPES[c] for c in self.sig.split('>')[0]] if self.sig else [float] * self.arity

    @property
    def ret(self):
        return TYPES[self.sig[-1]] if self.sig else float


class Grammar:
    # types — типы переменных ('SVV'); векторная переменная v — поле с компонентами vx, vy, vz
    def __init__(self, nodes, variables, types=None, ret=float):
        self.nodes, self.variables = nodes, variables
        vtypes = [TYPES[c] for c in types or 'S' * len(variables)]
        self.rows = ['*'] + [n.name for n in nodes]
        self.cols = [n.name for n in nodes] + variables
        self.ins = [[ret]] + [n.ins for n in nodes]  # слоты аргументов каждой строки M
        self.types = [n.ret for n in nodes] + vtypes  # тип каждого столбца M
        self.symbols = [sp.Matrix([Field(v + x) for x in AXES]) if t is Vec else Field(v)
                        for v, t in zip(variables, vtypes)]
        self.params = {p: sp.Symbol(p) for n in nodes for p, *_ in n.params}
        self.prior = {self.params[p]: (m, s) for n in nodes for p, m, s in n.params}
        self.pset = gp.PrimitiveSetTyped('F', vtypes, ret)
        self.pset.renameArguments(**{f'ARG{i}': v for i, v in enumerate(variables)})
        for n in nodes:
            w = [self.params[p] for p, *_ in n.params]
            self.pset.addPrimitive(KINDS[n.kind](w), n.ins, n.ret, name=n.name)

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
            # дети раскладываются по слотам по типу, дети одного типа — в порядке столбцов
            kids, args = list(np.flatnonzero(M[r])), []
            for t in self.ins[r]:
                j = next((j for j in kids if self.types[j] is t), None)
                if j is None or depth > len(self.rows):
                    raise ValueError(f'row {self.rows[r]}: no {t.__name__} argument')
                kids.remove(j)
                args.append(j)
            if kids:
                raise ValueError(f'row {self.rows[r]}: {len(kids)} extra args')
            if r:
                out.append(self.pset.mapping[self.rows[r]])
            for c in args:
                if c < nf:
                    build(c + 1, depth + 1)
                else:
                    out.append(self.pset.mapping[self.cols[c]])

        build(0)
        return gp.PrimitiveTree(out)

    def to_sympy(self, tree):
        r = gp.compile(tree, self.pset)(*self.symbols)
        return sp.Add(*[e * c for e, c in zip(E, r)]) if isinstance(r, sp.MatrixBase) else r

    def evaluate(self, tree, data, values):
        # data — {символ переменной: массив}; values — {символ параметра: число}
        expr = self.to_sympy(tree).subs(values)
        xs = sorted(expr.free_symbols, key=str)
        return sp.lambdify(xs, expr, 'numpy')(*[data[x] for x in xs])

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
