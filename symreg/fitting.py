# Fit formula parameters with variable projection (least squares + LM) or LM for all parameters.
import numpy as np
import sympy as sp
from scipy.optimize import least_squares


def nonlinear_params(expr, params):
    # A parameter is nonlinear if it appears inside an argument to a function such as sin.
    inside = set().union(*(f.free_symbols for f in expr.atoms(sp.Function)))
    return sorted(inside & set(params), key=str)


def linearize(expr, variables, theta):
    # f = sum_k c_k * phi_k(x; theta), where c_k are polynomials in the linear parameters.
    groups = {}
    for term in sp.Add.make_args(sp.expand(expr)):
        coef, basis = term.as_independent(*variables, *theta, as_Add=False)
        groups[basis] = groups.get(basis, 0) + coef
    return list(groups), list(groups.values())


def canonical(expr, params):
    # Model equivalence class up to the values of linear coefficients.
    variables = expr.free_symbols - set(params)
    return frozenset(linearize(expr, variables, nonlinear_params(expr, params))[0])


def _columns(fs, args, shape):
    return np.column_stack([np.broadcast_to(np.asarray(v, float), shape).ravel() for v in fs(*args)])


def linear_params(expr, params):
    # Greedily select jointly linear parameters in node order.
    # Each monomial contains at most one selected parameter, to the first power.
    nonlin, lin = nonlinear_params(expr, params), []
    terms = [{b: d for b, d in t.as_powers_dict().items() if b in params} for t in sp.Add.make_args(sp.expand(expr))]
    for p in params:
        if p not in nonlin and all(t.get(p, 0) <= 1 and not (p in t and any(q in t for q in lin)) for t in terms):
            lin.append(p)
    return lin


def fit(expr, data, y, prior, method='ls+lm', ks=np.linspace(-3, 3, 7)):
    # data maps variable symbols to arrays. prior maps parameter symbols to (mean, standard deviation).
    # ls+lm fits the jointly linear parameters by least squares with the other parameters fixed and optimizes those with LM.
    # Use mean + k*std starts for parameters inside functions. Return (NRMSE, parameter values) for the best start.
    # Parameters multiplying identically zero terms such as dot(cross(v, B), v) vanish after expansion.
    # They have no effect on the function and can interfere with the joint-linearity check before expansion.
    expr = expr.subs({p: 0 for p in expr.free_symbols - sp.expand(expr).free_symbols if p in prior})
    params = [p for p in prior if p in expr.free_symbols]
    variables = sorted(expr.free_symbols - set(params), key=str)
    data = [data[v] for v in variables]
    nonlin = nonlinear_params(expr, params)
    shape, y = np.shape(y), np.ravel(y)
    if method == 'ls+lm':
        lin = linear_params(expr, params)
        theta = [p for p in params if p not in lin]
        # f = f0(θ) + Σ_p p · ∂f/∂p(θ), for p in lin.
        phi = sp.lambdify([*variables, *theta], [expr.subs({p: 0 for p in lin}), *(sp.diff(expr, p) for p in lin)], 'numpy')

        def solve(th):
            A = _columns(phi, [*data, *th], shape)
            c = np.linalg.lstsq(A[:, 1:], y - A[:, 0], rcond=None)[0]
            return A[:, 1:] @ c + A[:, 0] - y, c

        def run(start):
            th = [start[p] for p in theta]
            if theta:
                th = least_squares(lambda th: solve(th)[0], th, method='lm').x
            r, c = solve(th)
            return r, dict(zip(theta, th)) | dict(zip(lin, c))
    elif method == 'lm':
        f = sp.lambdify([*variables, *params], expr, 'numpy')
        model = lambda p: np.broadcast_to(f(*data, *p), shape).ravel() - y

        def run(start):
            p = [start[q] for q in params]
            if params:
                p = least_squares(model, p, method='lm').x
            return model(p), dict(zip(params, p))
    else:
        raise ValueError(method)
    best = None
    for k in ks if nonlin else [0]:
        start = {p: prior[p][0] + (k * prior[p][1] if p in nonlin else 0) for p in params}
        r, values = run(start)
        err = float(np.sqrt(np.mean(r ** 2)) / np.std(y))
        if best is None or err < best[0]:
            best = err, values
    return best
