# Общий сценарий эксперимента для demos/vdp.py, lv.py, mhd.py: параметры по умолчанию задаёт файл модели,
# их можно переопределить из командной строки; --quick — быстрый прогон в figures/quick, не затирая результаты.
import argparse
from pathlib import Path

import numpy as np

from symreg import experiment
from symreg.experiment import Experiment, table
from symreg.plots import plot_accuracy, plot_phase, plot_tree


def run_model(name, repeats, trials, alphas):
    ap = argparse.ArgumentParser(description=f'{name}: noise in M -> tree recovery -> accuracy vs noise level')
    ap.add_argument('--repeats', type=int, default=repeats, help='new parameters and data')
    ap.add_argument('--trials', type=int, default=trials, help='noise realizations per repeat')
    ap.add_argument('--alphas', type=int, default=alphas, help='noise levels in [0, 1]')
    ap.add_argument('--quick', action='store_true', help='2 x 2 x 11, results go to figures/quick')
    args = ap.parse_args()
    if args.quick:
        args.repeats, args.trials, args.alphas = 2, 2, 11
        experiment.FIG = Path('figures/quick')
    experiment.FIG.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(1)
    exp = Experiment(name)
    print(exp.describe())
    if 'ode' in exp.model:
        plot_phase(exp, rng)
    plot_tree(exp, rng)

    alphas = np.linspace(0, 1, args.alphas)
    for extra in (False, True):
        exp = Experiment(name, extra)
        acc, zss, invalid = exp.run(args.repeats, args.trials, alphas)
        exp.save(alphas, acc, zss)
        print(f'{exp.dict}: invalid reconstructions: {invalid}')
        for p in acc:
            plot_accuracy(name, f'{exp.dict}_{p}', alphas, acc[p], zss[p])
            print(exp.dict, p)
            print(table(alphas, acc[p]))
    print('figures and .npz:', experiment.FIG)
