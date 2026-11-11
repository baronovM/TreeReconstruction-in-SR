# Дерево формулы: матрица смежности M, SymPy-выражение, поддеревья, расстояние zss.  python -m demos.trees
from symreg.experiment import Experiment
from symreg.fitting import canonical
from symreg.trees import subtree, swap_subtrees

eq = Experiment('vdp', extra=True).eqs[0]  # словарь с лишними узлами aff2, mul2, sq2
G, tree = eq.G, eq.tree
print('formula:', tree)
print('sympy:  ', eq.expr)
print('rows of M:', G.rows)
print('cols of M:', G.cols)
print(eq.M)
print('back from M:', G.from_matrix(eq.M))

sub = subtree(tree, 2)
print('\nsubtree at node 2:', sub, '=', G.to_sympy(sub))

# то же выражение через другие узлы словаря: другая M, ненулевой zss, но SymPy-эквивалентность
twin = G.parse('add3(mul(aff(sq2(x)), dx), sin(t), x)')
print('\ntwin:', twin)
print('zss distance:', G.zss_distance(tree, twin))
print('same canonical basis:', canonical(G.to_sympy(twin), G.params.values()) == eq.key)

a, b = G.parse(str(tree)), G.parse('add3(mul2(x, dx), sin(t), sq2(x))')
swap_subtrees(a, 2, b, 2)  # aff(sq(x)) <-> x
print('\nafter swap:', a, '|', b)
