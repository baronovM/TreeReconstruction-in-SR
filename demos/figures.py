# Перерисовать все картинки по сохранённым .npz без прогона экспериментов.  python -m demos.figures
import numpy as np

from symreg.experiment import FIG, Experiment, load
from symreg.models import MODELS
from symreg.plots import plot_accuracy, plot_phase, plot_summary, plot_tree

for name in MODELS:
    rng = np.random.default_rng(1)
    exp = Experiment(name)
    if 'ic' in exp.model:
        plot_phase(exp, rng)
    plot_tree(exp, rng)
    for f in FIG.glob(f'{name}_results_*.npz'):
        tag = f.stem.split('_results_')[1]
        alphas, methods, zss, acc = load(name, tag)
        plot_accuracy(name, tag, alphas, acc, zss, methods)
plot_summary()
