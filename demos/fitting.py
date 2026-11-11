# Подбор параметров по данным: LS+LM против LM, локальные минимумы по omega.  python -m demos.fitting
import numpy as np

from symreg.experiment import Experiment
from symreg.fitting import fit, linear_params, nonlinear_params

exp = Experiment('vdp')
eq, G = exp.eqs[0], exp.eqs[0].G
rng = np.random.default_rng(0)
values = exp.sample_params(rng)
data, (y,) = exp.simulate(values, rng)  # данные точные, шум только в M
params = list(G.params.values())
print('model:', eq.expr)
print('nonlinear:', nonlinear_params(eq.expr, params), '| least squares over:', linear_params(eq.expr, params))
mu, a1 = G.params['mu'], G.params['a1']
show = lambda v: {p.name: round(float(v[p]), 4) for p in params} | {'mu*a1': round(float(v[mu] * v[a1]), 4)}
print('  true:', show(values))
for method in ('ls+lm', 'lm'):
    err, found = fit(eq.expr, data, y, G.prior, method)
    print(f'{method:>6}:', show(found), f'NRMSE={err:.1e}')
print('mu, a0, a1 are identifiable only as products mu*a0, mu*a1: different values, same function')

wrong = G.parse('add3(mul(aff(sq(dx)), x), sin(t), x)')
print('\nwrong tree', wrong, f'NRMSE={fit(G.to_sympy(wrong), data, y, G.prior)[0]:.2f}')

# omega фиксирован, остальное — МНК: функционал ошибки по omega имеет минимумы с шагом ≈ pi/T
omega = G.params['omega']
grid = np.arange(0.45, 0.80, 0.0025)
errs = np.array([fit(eq.expr.subs(omega, w), data, y, G.prior)[0] for w in grid])
minima = grid[1:-1][(errs[1:-1] < errs[:-2]) & (errs[1:-1] < errs[2:])]
print(f'\nlocal minima in omega (true {values[omega]:.3f}):', minima.round(3).tolist())
