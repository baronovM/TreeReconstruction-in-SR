# Картинки: точность от уровня шума, сводка, фазовый портрет, M и деревья. Всё сохраняется в experiment.FIG.
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sympy as sp
from deap import gp

from . import experiment as ex
from .experiment import load, noisy
from .models import MODELS
from .recovery import METHODS, restore
from .trees import log_likelihood

plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
                     'font.size': 16, 'axes.titlesize': 17, 'legend.fontsize': 14})
TITLES = {'exact': r'exact match $\hat{M} = M$', 'sympy': 'symbolic equivalence', 'zss=0': 'zero tree edit distance',
          'fit ls+lm': 'data fit, LS + LM', 'fit lm': 'data fit, LM', 'zss': 'mean tree edit distance'}
label = lambda m: m.upper().replace('KMST', 'k-MST').replace('_', '+').replace('PRIM', 'Prim')
TEX = {'dx': r'\dot{x}', 'ddx': r'\ddot{x}', 'dy': r'\dot{y}', 'v': r'\mathbf{v}', 'B': r'\mathbf{B}',
       'drho': r'\partial_t \rho', 'dm': r'\partial_t (\rho \mathbf{v})', 'dB': r'\partial_t \mathbf{B}'}
tex = lambda v: f'${TEX.get(v, sp.latex(sp.Symbol(v)))}$'


def grid(ax):
    ax.minorticks_on()
    ax.grid(which='major', alpha=0.4)
    ax.grid(which='minor', alpha=0.12, lw=0.5)


def plot_accuracy(name, tag, alphas, acc, zss_mean, methods=METHODS):
    # tag = '<словарь>_<часть>'; тонкие линии — повторы, жирная — среднее
    panels = list(acc.items()) + [('zss', zss_mean)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True)
    colors = plt.cm.tab10.colors
    for ax, (crit, res) in zip(axes.flat, panels):
        for a, method in enumerate(methods):
            ls = '--' if 'kmst' in method else '-'
            for noodle in res[:, a]:
                ax.plot(alphas, noodle, color=colors[a], ls=ls, alpha=0.15, lw=0.8)
            ax.plot(alphas, res[:, a].mean(0), color=colors[a], ls=ls, lw=2.5, label=label(method))
        ax.set_title(TITLES[crit])
        grid(ax)
    for ax in axes[1]:
        ax.set_xlabel(r'noise level $\alpha$')
    for ax in axes[:, 0]:
        ax.set_ylabel('accuracy')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=len(methods), frameon=False)
    fig.suptitle(f'{MODELS[name]["title"]}: {tag.replace("_", " ")}')
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(ex.FIG / f'{name}_accuracy_{tag}.png', dpi=150)
    plt.close(fig)


def plot_summary(parts=(('vdp', ''), ('lv', '_system'), ('mhd', '_system')), method='prim'):
    # сводка: алгоритм Прима, три критерия, точный (сплошная) и расширенный (пунктир) словари
    fig, axes = plt.subplots(1, len(parts), figsize=(18, 6), sharey=True)
    for ax, (name, part) in zip(axes, parts):
        for tag, ls in [('exact_dict', '-'), ('extra_dict', '--')]:
            alphas, methods, _, acc = load(name, tag + part)
            a = list(methods).index(method)
            for i, (c, crit) in enumerate(zip(plt.cm.tab10.colors, ['exact', 'sympy', 'fit ls+lm'])):
                ax.plot(alphas, acc[crit][:, a].mean(0), color=c, ls=ls, lw=5 - 1.5 * i,
                        label=f'{TITLES[crit]}, {tag.replace("_", " ")}')
        ax.set(title=MODELS[name]['title'], xlabel=r'noise level $\alpha$')
        grid(ax)
    axes[0].set_ylabel('accuracy')
    fig.legend(*axes[0].get_legend_handles_labels(), loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(ex.FIG / 'summary.png', dpi=150)
    plt.close(fig)


def plot_phase(exp, rng):
    # только для моделей с ОДУ: фазовый портрет при средних параметрах и траектории при случайных
    label = lambda v: ', '.join(f'${sp.latex(p)}={v[p]:.3g}$' for p, (_, s) in exp.prior.items() if s > 0)
    mean = {p: m for p, (m, _) in exp.prior.items()}
    u, w = exp.model['variables'][:2]
    x = exp.eqs[0].G.symbols
    data, _ = exp.simulate(mean, rng, T=400, n=40000)
    us, ws = data[x[0]], data[x[1]]
    keep = np.linspace(0, 400, 40000) > 100
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.5))
    a1.plot(us[keep], ws[keep], lw=0.4, label=f'prior means: {label(mean)}; $t>100$')
    a1.set(xlabel=tex(u), ylabel=tex(w), title=f'{exp.model["title"]}: phase portrait')
    for _ in range(3):
        values = exp.sample_params(rng)
        data, _ = exp.simulate(values, rng, T=60, n=3000)
        a2.plot(np.linspace(0, 60, 3000), data[x[0]], lw=0.8, label=label(values))
    a2.set(xlabel='$t$', ylabel=tex(u), title='Trajectories, parameters sampled from the prior')
    for a in (a1, a2):
        grid(a)
        a.legend(loc='upper left', fontsize=11)
    fig.tight_layout()
    fig.savefig(ex.FIG / f'{exp.name}_phase.png', dpi=150)
    plt.close(fig)


def draw_tree(ax, tree, title, variables):
    nodes, edges, labels = gp.graph(tree)
    labels = {k: tex(v) if v in variables else v for k, v in labels.items()}
    g = nx.DiGraph(edges)
    g.add_nodes_from(nodes)
    pos = nx.nx_pydot.graphviz_layout(g, prog='dot')
    nx.draw(g, pos, ax=ax, labels=labels, node_color='lightsteelblue', node_size=5000, font_size=24,
            font_family='serif', arrows=False)
    ax.margins(0.1)
    ax.set_title(title, fontsize=22)


def plot_tree(exp, rng, alpha=0.7, method='prim'):
    # для каждого уравнения: M, зашумлённая M с восстановленными рёбрами, истинное и восстановленное деревья
    for eq in exp.eqs:
        G, M = eq.G, eq.M
        P = noisy(M, rng.uniform(-1, 1, M.shape), alpha)
        Mr = restore(method, P, G.ins, G.types)
        fig, axes = plt.subplots(2, 2, figsize=(14, 15), height_ratios=(1, 1.4))
        for ax, (A, title) in zip(axes[0], [(M, 'M (true)'), (P, rf'noisy M, $\alpha$={alpha}')]):
            ax.imshow(A, cmap='Greys', vmin=0, vmax=1)
            ax.set_xticks(range(len(G.cols)), [tex(c) if c in G.variables else c for c in G.cols],
                          rotation=45 if len(G.cols) > 8 else 0)
            ax.set_yticks(range(len(G.rows)), G.rows)
            ax.tick_params(labelsize=22)
            ax.set_title(title, fontsize=22)
            for i, j in np.argwhere(Mr if A is P else M):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec='red', lw=2))
        draw_tree(axes[1, 0], eq.tree, f'true tree for {tex(eq.target)}', G.variables)
        s = eq.structure(Mr)
        if s:
            draw_tree(axes[1, 1], s['tree'], f'{method}: zss={s["zss"]:g}\nlogL={log_likelihood(Mr, P):.2f} '
                                              f'(true {log_likelihood(M, P):.2f})', G.variables)
        fig.tight_layout()
        fig.savefig(ex.FIG / f'{exp.name}_tree_{eq.target}.png', dpi=150)
        plt.close(fig)
