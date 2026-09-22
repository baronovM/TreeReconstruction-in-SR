# Восстановление дерева по матрице вероятностей: порт RestoreMatrix из opt_pcst.ipynb.
# столбец 0 — корень (всегда 0), затем функции, затем переменные.
import numpy as np

try:
    import pcst_fast
except ImportError:
    pcst_fast = None


def _greedy(P, ar, nf, eps, depth_first):
    A, used, queue = np.zeros(P.shape, int), {0}, [0]

    def pick(f):
        row = P[f].copy()
        row[list(used)] = 0
        best_var = row[nf:].argmax() + nf
        chosen = []
        for _ in range(ar[f]):
            j = row.argmax()
            if row[j] < eps:
                j = best_var
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


def _prim(P, ar, nf, eps):
    ar, A, used, edges, last = list(ar), np.zeros(P.shape, int), {0}, {}, 0
    while True:
        if last < nf:
            row = P[last].copy()
            row[list(used)] = 0
            edges.update({(last, j): row[j] for j in np.flatnonzero(row > eps)})
        if not edges:
            return A
        i, j = max(edges, key=edges.get)
        A[i, j] = 1
        if j < nf:
            used.add(j)
        last = j
        ar[i] -= 1
        del edges[(i, j)]
        edges = {e: w for e, w in edges.items() if e[1] not in used and not (ar[i] == 0 and e[0] == i)}


def _pcst(P, nf, prize, eps):
    P[:, 0] = 0
    F = P[:, :nf]
    F = (F + F.T) / 2
    np.fill_diagonal(F, 0)
    edges = np.argwhere(F > eps)
    costs = 1 - F[F > eps] + eps
    _, chosen = pcst_fast.pcst_fast(edges, np.full(nf, prize), costs, 0, 1, 'gw', 0)
    return edges[chosen]


def restore(method, P, arities, eps=1e-4, prize=0.5):
    # P: (1 + F) x (F + V) в раскладке trees.Grammar; возвращает 0/1-матрицу той же формы
    P = np.hstack([np.zeros((P.shape[0], 1)), P])
    ar, nf = [1] + list(arities), P.shape[0]
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
        A = _greedy(P, ar, nf, eps, True)
    elif base == 'bfs':
        A = _greedy(P, ar, nf, eps, False)
    elif base == 'prim':
        A = _prim(P, ar, nf, eps)
    else:
        raise ValueError(method)
    return A[:, 1:]


METHODS = ['dfs', 'bfs', 'prim'] + (['kmst', 'kmst_dfs', 'kmst_bfs', 'kmst_prim'] if pcst_fast else [])
