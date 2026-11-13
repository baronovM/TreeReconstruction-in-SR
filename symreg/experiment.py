# Модель -> M -> шум U(-a, a) + нормировка -> восстановление -> проверка.
# Система ОДУ разбивается на независимые уравнения: у каждой правой части своё дерево и своя M.
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from tqdm import tqdm

from .fitting import canonical, fit, nonlinear_params
from .models import EXTRA, MODELS, Fields
from .recovery import METHODS, restore
from .trees import Grammar

FIT_METHODS = ['ls+lm', 'lm']
FIT_TOL = 1e-3  # успех: NRMSE после переподбора параметров меньше порога
FIG = Path('figures')  # результаты (.npz) и картинки


class Equation:
    def __init__(self, target, formula, nodes, variables, types=None):
        root = next(n for n in nodes if formula.startswith(n.name + '('))
        self.target, self.formula, self.G = target, formula, Grammar(nodes, variables, types, root.ret)
        self.tree = self.G.parse(formula)
        self.M = self.G.to_matrix(self.tree)
        self.expr = self.G.to_sympy(self.tree)
        self.key = canonical(self.expr, self.G.params.values())
        assert str(self.G.from_matrix(self.M)) == formula
        self.cache = {}

    def structure(self, Mr):
        key = Mr.tobytes()
        if key not in self.cache:
            try:
                tree = self.G.from_matrix(Mr)
            except ValueError:
                self.cache[key] = None
            else:
                expr = self.G.to_sympy(tree)
                self.cache[key] = dict(tree=tree, expr=expr, zss=self.G.zss_distance(tree, self.tree),
                                       sympy=canonical(expr, self.G.params.values()) == self.key)
        return self.cache[key]


def noisy(M, U, alpha):
    N = M + alpha * U
    return (N - N.min()) / (N.max() - N.min())


class Experiment:
    # модель из MODELS со словарём: точным (только узлы модели) или расширенным (extra=True, + лишние узлы)
    def __init__(self, name, extra=False):
        self.name, self.model = name, MODELS[name]
        self.dict = 'extra_dict' if extra else 'exact_dict'
        m = self.model
        self.eqs = [Equation(t, f, nodes + m.get('extra', EXTRA) * extra, m['variables'], m.get('types'))
                    for t, f, nodes in m['equations']]
        self.prior = {p: v for eq in self.eqs for p, v in eq.G.prior.items()}
        self.parts = [eq.target for eq in self.eqs] + ['system'] * (len(self.eqs) > 1)

    def sample_params(self, rng):
        return {p: m + s * rng.standard_normal() for p, (m, s) in self.prior.items()}

    def simulate(self, values, rng, T=60.0, n=1500):
        # данные не шумим: цели — точные значения правых частей
        if 'ic' not in self.model:
            data = Fields(rng)
        else:
            symbols = self.eqs[0].G.symbols
            has_t = self.model['variables'][-1] == 't'
            args = lambda t, s: (*s, t) if has_t else (*s,)
            reference = self.model['rhs'](values) if 'rhs' in self.model else None
            if reference is not None:
                rhs = lambda t, s: reference(s, t)
            else:
                fs = [sp.lambdify([*symbols, *values], eq.expr, 'numpy') for eq in self.eqs]
                v = list(values.values())
                rhs = lambda t, s: self.model['ode'](s, [f(*args(t, s), *v) for f in fs])
            ts = np.linspace(0, T, n)
            sol = solve_ivp(rhs, (0, T), rng.uniform(*self.model['ic'], len(self.model['variables']) - has_t),
                            t_eval=ts, method='LSODA', rtol=1e-9, atol=1e-9)
            data = dict(zip(symbols, args(ts, sol.y)))
            if reference is not None:
                return data, list(reference(sol.y.T, ts))
        return data, [eq.G.evaluate(eq.tree, data, values) for eq in self.eqs]

    def run(self, repeats, trials, alphas, seed=0):
        # Возвращает acc[part][критерий] и zss[part] — массивы [повтор, алгоритм, alpha], и число невалидных деревьев.
        # Части: уравнения и, для систем, вся система (успех — все уравнения сразу, zss — сумма)
        rng = np.random.default_rng(seed)
        crit = ['exact', 'sympy', 'zss=0'] + [f'fit {m}' for m in FIT_METHODS]
        shape = (len(METHODS), len(alphas))
        acc = {p: np.zeros((repeats, len(crit), *shape)) for p in self.parts}
        zss_mean = {p: np.zeros((repeats, *shape)) for p in self.parts}
        invalid = 0
        for r in tqdm(range(repeats), desc=f'{self.name} {self.dict}'):
            data, ys = self.simulate(self.sample_params(rng), rng)
            fits = {}
            for _ in range(trials):
                hits, dists = [], []
                for eq, y in zip(self.eqs, ys):
                    U = rng.uniform(-1, 1, eq.M.shape)  # одна реализация шума, масштабируется alpha
                    h, d = np.zeros((len(crit), *shape)), np.zeros(shape)
                    for k, alpha in enumerate(alphas):
                        P = noisy(eq.M, U, alpha)
                        for a, method in enumerate(METHODS):
                            Mr = restore(method, P, eq.G.ins, eq.G.types)
                            s = eq.structure(Mr)
                            if s is None:
                                invalid += 1
                                d[a, k] = len(eq.tree)
                                continue
                            fitted = []
                            for fm in FIT_METHODS:
                                key = (eq.target, Mr.tobytes(), fm)
                                if key not in fits:
                                    fits[key] = fit(s['expr'], data, y, eq.G.prior, fm)[0]
                                fitted.append(fits[key] < FIT_TOL)
                            h[:, a, k] = [np.array_equal(Mr, eq.M), s['sympy'], s['zss'] == 0, *fitted]
                            d[a, k] = s['zss']
                    hits.append(h)
                    dists.append(d)
                for p, h, d in zip(self.parts, hits + [np.prod(hits, 0)], dists + [np.sum(dists, 0)]):
                    acc[p][r] += h
                    zss_mean[p][r] += d
        return ({p: dict(zip(crit, acc[p].swapaxes(0, 1) / trials)) for p in self.parts},
                {p: v / trials for p, v in zss_mean.items()}, invalid)

    def save(self, alphas, acc, zss_mean):
        FIG.mkdir(parents=True, exist_ok=True)
        for p in acc:
            np.savez(FIG / f'{self.name}_results_{self.dict}_{p}.npz', alphas=alphas, methods=METHODS,
                     zss=zss_mean[p], **{k.replace(' ', '_'): v for k, v in acc[p].items()})

    def describe(self):
        lines = []
        for eq in self.eqs:
            theta = nonlinear_params(eq.expr, eq.G.params.values())
            lines += [f'{eq.target} = {eq.formula}', f'sympy: {eq.expr}',
                      f'nonlinear params: {theta} | linear: {sorted(set(eq.G.params.values()) - set(theta), key=str)}',
                      f'canonical basis: {sorted(map(str, eq.key))}',
                      f'M rows {eq.G.rows} cols {eq.G.cols}', str(eq.M)]
        return '\n'.join(lines)


def load(name, tag):
    # tag = '<словарь>_<часть>'; возвращает alphas, methods, zss, acc[критерий]
    d = dict(np.load(FIG / f'{name}_results_{tag}.npz'))
    return d.pop('alphas'), d.pop('methods'), d.pop('zss'), {k.replace('_', ' '): v for k, v in d.items()}


def table(alphas, acc, methods=METHODS, every=8):
    # средняя по повторам точность на каждом критерии в нескольких точках alpha
    step = len(alphas) // every
    rows = ['alpha: ' + str(alphas[::step].round(2).tolist())]
    for c, v in acc.items():
        rows.append(f'{c:>10}: ' + str({m: v[:, a].mean(0)[::step].round(2).tolist() for a, m in enumerate(methods)}))
    return '\n'.join(rows)
