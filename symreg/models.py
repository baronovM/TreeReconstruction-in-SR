# Описания физических моделей и синтетические данные для моделей без ОДУ (МГД).
import numpy as np

from .trees import AXES, E, Node

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
