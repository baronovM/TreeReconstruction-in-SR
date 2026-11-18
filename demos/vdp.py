# Forced Van der Pol oscillator (about 6 min). Run with python -m demos.vdp [--quick].
from demos.common import run_model

run_model('vdp', repeats=20, trials=20, alphas=41)
