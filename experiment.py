# Модель -> M -> шум U(-a, a) + нормировка -> восстановление -> проверка.
# Система ОДУ разбивается на независимые уравнения: у каждой правой части своё дерево и своя M.
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sympy as sp
from deap import gp
from scipy.integrate import solve_ivp
from tqdm import tqdm

from fitting import canonical, fit, nonlinear_params
from recovery import METHODS, restore
from trees import AXES, E, Grammar, Node, log_likelihood

# лишние узлы словаря, которых нет в истинной модели
EXTRA = [Node('aff2', 'aff', 1, [('b0', 0.0, 0.0), ('b1', 1.0, 0.0)]), Node('mul2', 'mul', 2), Node('sq2', 'sq', 1)]
# equations: (цель, формула с аргументами в порядке столбцов M, узлы словаря); types — типы переменных;
# ode: производная состояния по состоянию s и значениям правых частей ys; 't' — последняя переменная.
# Модель без ode — поля в пространстве: данные — снимок случайных гладких полей (Fields)
MODELS = {
    # ẍ = mu (1 - x²) ẋ - x + A sin(omega t);  mu, A — линейные, omega — нелинейный
    'vdp': dict(title='Forced Van der Pol', variables=['x', 'dx', 't'], ic=(-2, 2),
                ode=lambda s, ys: [s[1], *ys],
                equations=[('ddx', 'add3(mul(aff(sq(x)), dx), sin(t), x)', [
                    Node('add3', 'lin', 3, [('mu', 8.53, 0.5), ('A', 1.2, 0.1), ('c_x', -1.0, 0.0)]),
                    Node('mul', 'mul', 2),
                    Node('aff', 'aff', 1, [('a0', 1.0, 0.0), ('a1', -1.0, 0.0)]),
                    Node('sq', 'sq', 1),
                    Node('sin', 'sin', 1, [('omega', 0.63, 0.02)]),
                ])]),
    # ẋ = x (alpha + beta y),  ẏ = y (gamma + delta x);  все параметры линейные, beta, gamma < 0
    'lv': dict(title='Lotka-Volterra', variables=['x', 'y'], ic=(1, 10),
               ode=lambda s, ys: ys,
               equations=[
                   ('dx', 'mul(aff(y), x)', [Node('mul', 'mul', 2),
                                              Node('aff', 'aff', 1, [('alpha', 1.1, 0.1), ('beta', -0.4, 0.04)])]),
                   ('dy', 'mul(aff(x), y)', [Node('mul', 'mul', 2),
                                              Node('aff', 'aff', 1, [('gamma', -0.4, 0.04), ('delta', 0.1, 0.01)])]),
               ]),
    # изотермическая МГД (The Well): ∂ρ/∂t = -∇·(ρv), ∂(ρv)/∂t = -∇·(ρvv - BB) - c²∇ρ, ∂B/∂t = ∇×(v×B)
    'mhd': dict(title='Isothermal MHD', variables=['rho', 'v', 'B'], types='SVV',
                equations=[
                    ('drho', 'div(mul(rho, v))', [Node('div', 'div', 1, [('k_rho', -1.0, 0.0)], 'V>S'),
                                                  Node('mul', 'mul', 2, sig='SV>V')]),
                    ('dm', 'add(divT(sub(outer(mul(rho, v), v), sq(B))), grad(rho))', [
                        Node('add', 'lin', 2, [('k_m', -1.0, 0.0), ('c_p', -1.0, 0.1)], 'VV>V'),
                        Node('divT', 'div', 1, sig='T>V'),
                        Node('sub', 'lin', 2, [('k_v', 1.0, 0.0), ('k_B', -1.0, 0.0)], 'TT>T'),
                        Node('outer', 'outer', 2, sig='VV>T'),
                        Node('mul', 'mul', 2, sig='SV>V'),
                        Node('sq', 'sq', 1, sig='V>T'),
                        Node('grad', 'grad', 1, sig='S>V'),
                    ]),
                    ('dB', 'curl(cross(v, B))', [Node('curl', 'curl', 1, sig='V>V'),
                                                 Node('cross', 'cross', 2, sig='VV>V')]),
                ],
                extra=[Node('grad2', 'grad', 1, sig='S>V'), Node('div2', 'div', 1, sig='V>S'),
                       Node('curl2', 'curl', 1, sig='V>V'), Node('cross2', 'cross', 2, sig='VV>V'),
                       Node('dot', 'dot', 2, sig='VV>S')]),
}
FIT_METHODS = ['ls+lm', 'lm']
FIT_TOL = 1e-3  # успех: NRMSE после переподбора параметров меньше порога
FIG = Path('figures')


class Equation:
    def __init__(self, target, formula, nodes, variables, types=None):
        root = next(n for n in nodes if formula.startswith(n.name + '('))
        self.target, self.G = target, Grammar(nodes, variables, types, root.ret)
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


def setup(name, extra=False):
    global NAME, MODEL, EQS, PRIOR
    NAME, MODEL = name, MODELS[name]
    EQS = [Equation(t, f, nodes + MODEL.get('extra', EXTRA) * extra, MODEL['variables'], MODEL.get('types'))
           for t, f, nodes in MODEL['equations']]
    PRIOR = {p: v for eq in EQS for p, v in eq.G.prior.items()}


def sample_params(rng):
    return {p: m + s * rng.standard_normal() for p, (m, s) in PRIOR.items()}


class Fields(dict):
    # Случайные гладкие периодические поля ρ = 1 + δρ, v, B (∇·B = 0) на сетке n³, гармоники |k| <= kmax.
    # Производные — по запросу, спектрально: data[Field('vx_y')] = ∂vx/∂y; e_i — индикаторы компонент
    def __init__(self, rng, n=12, kmax=3):
        k = np.fft.fftfreq(n, 1 / n)
        self.K = np.stack(np.meshgrid(k, k, k, indexing='ij'))
        low = np.linalg.norm(self.K, axis=0) <= kmax
        v, B, r = (np.fft.fftn(rng.standard_normal((3, n, n, n)), axes=(1, 2, 3)) * low for _ in range(3))
        B -= self.K * (self.K * B).sum(0) / np.maximum((self.K ** 2).sum(0), 1)
        real = lambda f: np.fft.ifftn(f, axes=(-3, -2, -1)).real
        rho = r[0] / (2 * np.abs(real(r[0])).max())  # |δρ| <= 0.5
        rho[0, 0, 0] += n ** 3
        self.hat = {'rho': rho} | {f'{name}{x}': f[i] / real(f).std()
                                    for name, f in [('v', v), ('B', B)] for i, x in enumerate(AXES)}
        super().__init__({e: np.eye(3)[:, [i]] for i, e in enumerate(E)})

    def __missing__(self, s):
        base, _, axes = s.name.partition('_')
        self[s] = np.fft.ifftn(self.hat[base] * np.prod([1j * self.K[AXES.index(a)] for a in axes], 0)).real.ravel()
        return self[s]


def simulate(values, rng, T=60.0, n=1500):
    if 'ode' not in MODEL:
        data = Fields(rng)
    else:
        fs = [sp.lambdify([*EQS[0].G.symbols, *values], eq.expr, 'numpy') for eq in EQS]
        v = list(values.values())
        has_t = MODEL['variables'][-1] == 't'
        args = lambda t, s: (*s, t) if has_t else (*s,)
        rhs = lambda t, s: MODEL['ode'](s, [f(*args(t, s), *v) for f in fs])
        ts = np.linspace(0, T, n)
        sol = solve_ivp(rhs, (0, T), rng.uniform(*MODEL['ic'], len(MODEL['variables']) - has_t),
                        t_eval=ts, method='LSODA', rtol=1e-9, atol=1e-9)
        data = dict(zip(EQS[0].G.symbols, args(ts, sol.y)))
    return data, [eq.G.evaluate(eq.tree, data, values) for eq in EQS]  # данные не шумим: цели — точные


def noisy(M, U, alpha):
    N = M + alpha * U
    return (N - N.min()) / (N.max() - N.min())


def run(repeats, trials, alphas, seed=0):
    # части: уравнения и, для систем, вся система (успех — все уравнения сразу, zss — сумма)
    rng = np.random.default_rng(seed)
    crit = ['exact', 'sympy', 'zss=0'] + [f'fit {m}' for m in FIT_METHODS]
    parts = [eq.target for eq in EQS] + ['system'] * (len(EQS) > 1)
    shape = (len(METHODS), len(alphas))
    acc = {p: np.zeros((repeats, len(crit), *shape)) for p in parts}
    zss_mean = {p: np.zeros((repeats, *shape)) for p in parts}
    invalid = 0
    for r in tqdm(range(repeats), desc='repeats'):
        data, ys = simulate(sample_params(rng), rng)
        fits = {}
        for _ in range(trials):
            hits, dists = [], []
            for eq, y in zip(EQS, ys):
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
            for p, h, d in zip(parts, hits + [np.prod(hits, 0)], dists + [np.sum(dists, 0)]):
                acc[p][r] += h
                zss_mean[p][r] += d
    print(f'invalid reconstructions: {invalid}')
    return {p: dict(zip(crit, acc[p].swapaxes(0, 1) / trials)) for p in parts}, \
           {p: v / trials for p, v in zss_mean.items()}


def plot_accuracy(alphas, acc, zss_mean, tag):
    panels = list(acc.items()) + [('mean zss distance', zss_mean)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    colors = plt.cm.tab10.colors
    for ax, (name, res) in zip(axes.flat, panels):
        for a, method in enumerate(METHODS):
            for noodle in res[:, a]:
                ax.plot(alphas, noodle, color=colors[a], alpha=0.15, lw=0.8)
            ax.plot(alphas, res[:, a].mean(0), color=colors[a], lw=2, label=method)
        ax.set_title(name)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel(r'noise level $\alpha$')
    axes[0, 0].set_ylabel('ratio of correct reconstructions')
    axes[0, 0].legend()
    fig.suptitle(f'{MODEL["title"]}: {tag}')
    fig.tight_layout()
    fig.savefig(FIG / f'{NAME}_accuracy_{tag}.png', dpi=150)


def plot_phase(rng):
    label = lambda v: ', '.join(f'${sp.latex(p)}={v[p]:.3g}$' for p, (_, s) in PRIOR.items() if s > 0)
    mean = {p: m for p, (m, _) in PRIOR.items()}
    u, w = MODEL['variables'][:2]
    x = EQS[0].G.symbols
    data, _ = simulate(mean, rng, T=400, n=40000)
    us, ws = data[x[0]], data[x[1]]
    keep = np.linspace(0, 400, 40000) > 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(us[keep], ws[keep], lw=0.4, label=f'prior means: {label(mean)}; $t>100$')
    a1.set(xlabel=f'${u}$', ylabel=f'${w}$', title=f'{MODEL["title"]}: phase portrait')
    for _ in range(3):
        values = sample_params(rng)
        data, _ = simulate(values, rng, T=60, n=3000)
        a2.plot(np.linspace(0, 60, 3000), data[x[0]], lw=0.8, label=label(values))
    a2.set(xlabel='$t$', ylabel=f'${u}$', title='Trajectories, parameters sampled from the prior')
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / f'{NAME}_phase.png', dpi=150)


def draw_tree(ax, tree, title):
    nodes, edges, labels = gp.graph(tree)
    g = nx.DiGraph(edges)
    g.add_nodes_from(nodes)
    pos = nx.nx_pydot.graphviz_layout(g, prog='dot')
    nx.draw(g, pos, ax=ax, labels=labels, node_color='lightsteelblue', node_size=1100, arrows=False)
    ax.set_title(title)


def plot_tree(rng, alpha=0.7, method='prim'):
    for eq in EQS:
        G, M = eq.G, eq.M
        P = noisy(M, rng.uniform(-1, 1, M.shape), alpha)
        Mr = restore(method, P, G.ins, G.types)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        for ax, (A, title) in zip(axes[0], [(M, 'M (true)'), (P, rf'noisy M, $\alpha$={alpha}')]):
            ax.imshow(A, cmap='Greys', vmin=0, vmax=1)
            ax.set(xticks=range(len(G.cols)), xticklabels=G.cols, yticks=range(len(G.rows)),
                   yticklabels=G.rows, title=title)
            for i, j in np.argwhere(Mr if A is P else M):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec='red', lw=2))
        draw_tree(axes[1, 0], eq.tree, f'true tree for {eq.target}')
        s = eq.structure(Mr)
        if s:
            draw_tree(axes[1, 1], s['tree'], f'{method}: zss={s["zss"]}, logL={log_likelihood(Mr, P):.2f} '
                                              f'(true {log_likelihood(M, P):.2f})')
        fig.tight_layout()
        fig.savefig(FIG / f'{NAME}_tree_{eq.target}.png', dpi=150)


def describe():
    for eq, (_, formula, _) in zip(EQS, MODEL['equations']):
        theta = nonlinear_params(eq.expr, eq.G.params.values())
        print(f'{eq.target} =', formula)
        print('sympy:', eq.expr)
        print('nonlinear params:', theta, '| linear:', sorted(set(eq.G.params.values()) - set(theta), key=str))
        print('canonical basis:', sorted(map(str, eq.key)))
        print('M rows', eq.G.rows, 'cols', eq.G.cols)
        print(eq.M)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('model', choices=MODELS)
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--alphas', type=int, default=21)
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    setup(args.model)
    describe()
    if 'ode' in MODEL:
        plot_phase(rng)
    plot_tree(rng)
    alphas = np.linspace(0, 1, args.alphas)
    for tag, extra in [('exact_dict', False), ('extra_dict', True)]:
        setup(args.model, extra)
        acc, zss_mean = run(args.repeats, args.trials, alphas)
        for p in acc:
            np.savez(FIG / f'{NAME}_results_{tag}_{p}.npz', alphas=alphas, methods=METHODS, zss=zss_mean[p],
                     **{k.replace(' ', '_'): v for k, v in acc[p].items()})
            plot_accuracy(alphas, acc[p], zss_mean[p], f'{tag}_{p}')
            print(tag, p, 'alpha:', alphas[::len(alphas) // 8].round(2).tolist())
            for c, v in acc[p].items():
                print(f'{c:>10}:', {m: v[:, a].mean(0)[::len(alphas) // 8].round(2).tolist() for a, m in enumerate(METHODS)})
