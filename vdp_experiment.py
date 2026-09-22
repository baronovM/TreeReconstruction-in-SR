# Осциллятор Ван дер Поля: модель -> M -> шум U(-a, a) + нормировка -> восстановление -> проверка.
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
from trees import Grammar, Node, log_likelihood

# ẍ = mu (1 - x²) ẋ - x + A sin(omega t);  mu, A — линейные, omega — нелинейный
VARIABLES = ['x', 'dx', 't']
NODES = [
    Node('add3', 'lin', 3, [('mu', 8.53, 0.5), ('A', 1.2, 0.1), ('c_x', -1.0, 0.0)]),
    Node('mul', 'mul', 2),
    Node('aff', 'aff', 1, [('a0', 1.0, 0.0), ('a1', -1.0, 0.0)]),
    Node('sq', 'sq', 1),
    Node('sin', 'sin', 1, [('omega', 0.63, 0.02)]),
]
# лишние узлы словаря, которых нет в истинной модели
EXTRA = [Node('aff2', 'aff', 1, [('b0', 0.0, 0.0), ('b1', 1.0, 0.0)]), Node('mul2', 'mul', 2), Node('sq2', 'sq', 1)]
FORMULA = 'add3(mul(aff(sq(x)), dx), sin(t), x)'  # аргументы в порядке столбцов M
FIT_METHODS = ['ls+lm', 'lm']
FIT_TOL = 1e-3  # успех: NRMSE после переподбора параметров меньше порога
FIG = Path('figures')


def sample_params(rng):
    return {p: m + s * rng.standard_normal() for p, (m, s) in G.prior.items()}


def simulate(values, rng, T=60.0, n=1500):
    f = sp.lambdify([*G.symbols, *values], TRUE_EXPR, 'numpy')
    v = list(values.values())
    ts = np.linspace(0, T, n)
    sol = solve_ivp(lambda t, s: [s[1], f(s[0], s[1], t, *v)], (0, T), rng.uniform(-2, 2, 2),
                    t_eval=ts, method='LSODA', rtol=1e-9, atol=1e-9)
    x, dx = sol.y
    return (x, dx, ts), f(x, dx, ts, *v)  # данные не шумим: цель — точное ẍ


def noisy(M, U, alpha):
    N = M + alpha * U
    return (N - N.min()) / (N.max() - N.min())


def structure(Mr):
    cache, key = structure.cache, Mr.tobytes()
    if key not in cache:
        try:
            tree = G.from_matrix(Mr)
        except ValueError:
            cache[key] = None
        else:
            expr = G.to_sympy(tree)
            cache[key] = dict(tree=tree, expr=expr, zss=G.zss_distance(tree, TRUE_TREE),
                              sympy=canonical(expr, G.symbols, G.params.values()) == TRUE_KEY)
    return cache[key]


def setup(nodes):
    global G, TRUE_TREE, M_TRUE, TRUE_EXPR, TRUE_KEY
    G = Grammar(nodes, VARIABLES)
    TRUE_TREE = G.parse(FORMULA)
    M_TRUE = G.to_matrix(TRUE_TREE)
    TRUE_EXPR = G.to_sympy(TRUE_TREE)
    TRUE_KEY = canonical(TRUE_EXPR, G.symbols, G.params.values())
    assert str(G.from_matrix(M_TRUE)) == FORMULA
    structure.cache = {}


setup(NODES)


def run(repeats, trials, alphas, seed=0):
    rng = np.random.default_rng(seed)
    crit = ['exact', 'sympy', 'zss=0'] + [f'fit {m}' for m in FIT_METHODS]
    acc = {c: np.zeros((repeats, len(METHODS), len(alphas))) for c in crit}
    zss_mean = np.zeros((repeats, len(METHODS), len(alphas)))
    invalid = 0
    for r in tqdm(range(repeats), desc='repeats'):
        data, y = simulate(sample_params(rng), rng)
        fits = {}
        for _ in range(trials):
            U = rng.uniform(-1, 1, M_TRUE.shape)  # одна реализация шума, масштабируется alpha
            for k, alpha in enumerate(alphas):
                P = noisy(M_TRUE, U, alpha)
                for a, method in enumerate(METHODS):
                    Mr = restore(method, P, G.arities)
                    s = structure(Mr)
                    if s is None:
                        invalid += 1
                        zss_mean[r, a, k] += len(TRUE_TREE) / trials
                        continue
                    acc['exact'][r, a, k] += np.array_equal(Mr, M_TRUE)
                    acc['sympy'][r, a, k] += s['sympy']
                    acc['zss=0'][r, a, k] += s['zss'] == 0
                    zss_mean[r, a, k] += s['zss'] / trials
                    for fm in FIT_METHODS:
                        key = (Mr.tobytes(), fm)
                        if key not in fits:
                            fits[key] = fit(s['expr'], G.symbols, data, y, G.prior, fm)[0]
                        acc[f'fit {fm}'][r, a, k] += fits[key] < FIT_TOL
    print(f'invalid reconstructions: {invalid}')
    return {c: v / trials for c, v in acc.items()}, zss_mean


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
    fig.suptitle(tag)
    fig.tight_layout()
    fig.savefig(FIG / f'vdp_accuracy_{tag}.png', dpi=150)


def plot_phase(rng):
    label = lambda v: ', '.join(f'${sp.latex(p)}={v[p]:.3g}$' for p, (_, s) in G.prior.items() if s > 0)
    mean = {p: m for p, (m, _) in G.prior.items()}
    (x, dx, t), _ = simulate(mean, rng, T=400, n=40000)
    keep = t > 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    a1.plot(x[keep], dx[keep], lw=0.4, label=f'prior means: {label(mean)}; $t>100$')
    a1.set(xlabel='$x$', ylabel=r'$\dot x$', title='Forced Van der Pol: phase portrait')
    for _ in range(3):
        values = sample_params(rng)
        (xs, dxs, _), _ = simulate(values, rng, T=60, n=3000)
        a2.plot(np.linspace(0, 60, 3000), xs, lw=0.8, label=label(values))
    a2.set(xlabel='$t$', ylabel='$x$', title='Trajectories, parameters sampled from the prior')
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend(loc='upper left', fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / 'vdp_phase.png', dpi=150)


def draw_tree(ax, tree, title):
    nodes, edges, labels = gp.graph(tree)
    g = nx.DiGraph(edges)
    g.add_nodes_from(nodes)
    pos = nx.nx_pydot.graphviz_layout(g, prog='dot')
    nx.draw(g, pos, ax=ax, labels=labels, node_color='lightsteelblue', node_size=1100, arrows=False)
    ax.set_title(title)


def plot_tree(rng, alpha=0.7, method='prim'):
    P = noisy(M_TRUE, rng.uniform(-1, 1, M_TRUE.shape), alpha)
    Mr = restore(method, P, G.arities)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (A, title) in zip(axes[0], [(M_TRUE, 'M (true)'), (P, rf'noisy M, $\alpha$={alpha}')]):
        ax.imshow(A, cmap='Greys', vmin=0, vmax=1)
        ax.set(xticks=range(len(G.cols)), xticklabels=G.cols, yticks=range(len(G.rows)),
               yticklabels=G.rows, title=title)
        for i, j in np.argwhere(Mr if A is P else M_TRUE):
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec='red', lw=2))
    draw_tree(axes[1, 0], TRUE_TREE, 'true tree')
    s = structure(Mr)
    if s:
        draw_tree(axes[1, 1], s['tree'], f'{method}: zss={s["zss"]}, logL={log_likelihood(Mr, P):.2f} '
                                          f'(true {log_likelihood(M_TRUE, P):.2f})')
    fig.tight_layout()
    fig.savefig(FIG / 'vdp_tree.png', dpi=150)


def describe():
    theta = nonlinear_params(TRUE_EXPR, G.params.values())
    print('model:', FORMULA)
    print('sympy:', TRUE_EXPR)
    print('nonlinear params:', theta, '| linear:', sorted(set(G.params.values()) - set(theta), key=str))
    print('canonical basis:', sorted(map(str, TRUE_KEY)))
    print('M rows', G.rows, 'cols', G.cols)
    print(M_TRUE)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=10)
    ap.add_argument('--trials', type=int, default=20)
    ap.add_argument('--alphas', type=int, default=21)
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    describe()
    plot_phase(rng)
    plot_tree(rng)
    alphas = np.linspace(0, 1, args.alphas)
    for tag, nodes in [('exact_dict', NODES), ('extra_dict', NODES + EXTRA)]:
        setup(nodes)
        acc, zss_mean = run(args.repeats, args.trials, alphas)
        np.savez(FIG / f'vdp_results_{tag}.npz', alphas=alphas, methods=METHODS, zss=zss_mean,
                 **{k.replace(' ', '_'): v for k, v in acc.items()})
        plot_accuracy(alphas, acc, zss_mean, tag)
        print(tag, 'alpha:', alphas[::len(alphas) // 8].round(2).tolist())
        for c, v in acc.items():
            print(f'{c:>10}:', {m: v[:, a].mean(0)[::len(alphas) // 8].round(2).tolist() for a, m in enumerate(METHODS)})
