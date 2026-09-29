# Подбор параметров формулы: МНК для линейных + LM для нелинейных (variable projection) или LM сразу на всё.
import numpy as np
import sympy as sp
from scipy.optimize import least_squares


def nonlinear_params(expr, params):
    # параметр нелинейный, если он стоит внутри аргумента какой-либо функции (sin и т.п.)
    inside = set().union(*(f.free_symbols for f in expr.atoms(sp.Function)))
    return sorted(inside & set(params), key=str)


def linearize(expr, variables, theta):
    # f = sum_k c_k * phi_k(x; theta): коэффициенты c_k — полиномы от линейных параметров
    groups = {}
    for term in sp.Add.make_args(sp.expand(expr)):
        coef, basis = term.as_independent(*variables, *theta, as_Add=False)
        groups[basis] = groups.get(basis, 0) + coef
    return list(groups), list(groups.values())


def canonical(expr, params):
    # класс эквивалентности модели с точностью до значений линейных коэффициентов
    variables = expr.free_symbols - set(params)
    return frozenset(linearize(expr, variables, nonlinear_params(expr, params))[0])


def _columns(fs, args, shape):
    return np.column_stack([np.broadcast_to(np.asarray(v, float), shape).ravel() for v in fs(*args)])


def fit(expr, data, y, prior, method='ls+lm', ks=np.linspace(-3, 3, 7), max_basis=0.1):
    # data: {символ переменной: массив}; prior: {символ параметра: (матожидание, ст.откл.)}.
    # Мультистарт: нелинейные параметры = mean + k*std. Возвращает (NRMSE, значения параметров) лучшего старта.
    # ls+lm с базисом больше max_basis * (число наблюдений) — неудача: такой МНК подгоняет что угодно
    params = sorted(expr.free_symbols & set(prior), key=str)
    variables = sorted(expr.free_symbols - set(params), key=str)
    data = [data[v] for v in variables]
    theta = nonlinear_params(expr, params)
    shape, y = np.shape(y), np.ravel(y)
    if method == 'ls+lm':
        basis, _ = linearize(expr, variables, theta)
        if len(basis) > max_basis * len(y):
            return np.inf, {}
        phi = sp.lambdify([*variables, *theta], basis, 'numpy')

        def solve(th):
            A = _columns(phi, [*data, *th], shape)
            c = np.linalg.lstsq(A, y, rcond=None)[0]
            return A @ c - y, c

        def run(start):
            th = [start[p] for p in theta]
            if theta:
                th = least_squares(lambda th: solve(th)[0], th, method='lm').x
            r, c = solve(th)
            return r, dict(zip(theta, th)) | dict(zip(basis, c))
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
    for k in ks if theta else [0]:
        start = {p: prior[p][0] + (k * prior[p][1] if p in theta else 0) for p in params}
        r, values = run(start)
        err = float(np.sqrt(np.mean(r ** 2)) / np.std(y))
        if best is None or err < best[0]:
            best = err, values
    return best
