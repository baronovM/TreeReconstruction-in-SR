# Two-equation Lotka–Volterra system. Run with python -m demos.lv [--quick].
from demos.common import run_model

run_model('lv', repeats=20, trials=20, alphas=41)
