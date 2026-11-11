# Изотермическая МГД, типизированные деревья; ≈40 мин из-за расширенного словаря.  python -m demos.mhd  [--quick]
from demos.common import run_model

run_model('mhd', repeats=10, trials=10, alphas=21)
