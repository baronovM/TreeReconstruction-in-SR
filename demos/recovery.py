# Perturb M and recover the tree with different algorithms. Run with python -m demos.recovery.
import numpy as np

from symreg.experiment import Experiment, noisy
from symreg.recovery import METHODS, restore
from symreg.trees import log_likelihood

eq = Experiment('vdp', extra=True).eqs[0]
G, M = eq.G, eq.M
U = np.random.default_rng(0).uniform(-1, 1, M.shape)  # Reuse one noise draw for all alpha values.
print('true:', eq.tree)
for alpha in (0.3, 0.6, 0.9):
    P = noisy(M, U, alpha)  # P = minmax(M + alpha U)
    print(f'\nalpha = {alpha}: logL(true) = {log_likelihood(M, P):.2f}')
    for method in METHODS:
        Mr = restore(method, P, G.ins, G.types)
        s = eq.structure(Mr)
        if s is None:
            print(f'{method:>9}: invalid tree')
        else:
            print(f'{method:>9}: exact={np.array_equal(Mr, M):d} sympy={s["sympy"]:d} zss={s["zss"]} '
                  f'logL={log_likelihood(Mr, P):.2f}  {s["tree"]}')
