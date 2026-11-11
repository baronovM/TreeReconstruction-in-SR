# Восстановление дерева по матрице вероятностей: порт RestoreMatrix из opt_pcst.ipynb.
# столбец 0 — корень (всегда 0), затем функции, затем переменные.
import numpy as np

try:
    import pcst_fast
except ImportError:
    pcst_fast = None


# ins[i] — типы слотов аргументов строки i, of[t] — маска столбцов типа t
def _greedy(P, ins, of, nf, eps, depth_first):
    A, used, queue = np.zeros(P.shape, int), {0}, [0]

    def pick(f):
        row = P[f].copy()
        row[list(used)] = 0
        chosen = []
        for t in ins[f]:
            j = np.where(of[t], row, -1).argmax()
            if row[j] < eps or not of[t][j]:
                var = np.where(of[t][nf:], P[f, nf:], -1)
                if var.max() < 0:
                    continue
                j = var.argmax() + nf
            row[j], A[f, j] = 0, 1
            chosen.append(j)
        return [j for j in chosen if j < nf]

    if depth_first:
        def rec(f):
            kids = pick(f)
            used.update(kids)
            for j in kids:
                rec(j)
        rec(0)
    else:
        while queue:
            kids = pick(queue.pop(0))
            used.update(kids)
            queue += kids
    return A


def _prim(P, ins, of, types, nf, eps):
    left, A, used, edges, last = [list(s) for s in ins], np.zeros(P.shape, int), {0}, {}, 0
    while True:
        if last < nf:
            row = P[last].copy()
            row[list(used)] = 0
            ok = np.any([of[t] for t in left[last]], 0)
            edges.update({(last, j): row[j] for j in np.flatnonzero((row > eps) & ok)})
        if not edges:
            return A
        i, j = max(edges, key=edges.get)
        A[i, j] = 1
        if j < nf:
            used.add(j)
        last = j
        left[i].remove(types[j])
        del edges[(i, j)]
        edges = {e: w for e, w in edges.items() if e[1] not in used and types[e[1]] in left[e[0]]}


def _pcst(P, nf, prize, eps):
    P[:, 0] = 0
    F = P[:, :nf]
    F = (F + F.T) / 2
    np.fill_diagonal(F, 0)
    edges = np.argwhere(F > eps)
    costs = 1 - F[F > eps] + eps
    _, chosen = pcst_fast.pcst_fast(edges, np.full(nf, prize), costs, 0, 1, 'gw', 0)
    return edges[chosen]


def restore(method, P, ins, types, eps=1e-4, prize=0.5):
    # P: (1 + F) x (F + V) в раскладке trees.Grammar; ins, types — Grammar.ins, Grammar.types.
    # Слот типа t заполняется только столбцом типа t. Возвращает 0/1-матрицу той же формы, что P
    P = np.hstack([np.zeros((P.shape[0], 1)), P])
    nf, types = P.shape[0], [None] + list(types)
    of = {t: np.array([c is t for c in types]) for s in ins for t in s}
    base = method
    if method.startswith('kmst'):
        if pcst_fast is None:
            raise ImportError('pcst_fast is not installed')
        chosen = _pcst(P, nf, prize, eps)
        if method == 'kmst':
            P[:, :nf] = 0
            P[chosen[:, 0], chosen[:, 1]] = P[chosen[:, 1], chosen[:, 0]] = 1
            base = 'dfs'
        else:
            for i, j in chosen:
                P[i, j] = (P[i, j] + 1) / 2
                P[j, i] = (P[j, i] + 1) / 2
            base = method.split('_')[1]
    if base == 'dfs':
        A = _greedy(P, ins, of, nf, eps, True)
    elif base == 'bfs':
        A = _greedy(P, ins, of, nf, eps, False)
    elif base == 'prim':
        A = _prim(P, ins, of, types, nf, eps)
    else:
        raise ValueError(method)
    return A[:, 1:]


METHODS = ['dfs', 'bfs', 'prim'] + (['kmst', 'kmst_dfs', 'kmst_bfs', 'kmst_prim'] if pcst_fast else [])
