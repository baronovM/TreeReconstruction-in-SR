# Isothermal MHD with typed trees (about 40 min due to the extended dictionary). Run with python -m demos.mhd [--quick].
from demos.common import run_model

run_model('mhd', repeats=10, trials=10, alphas=21)
