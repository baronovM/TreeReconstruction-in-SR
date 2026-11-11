# Вынужденный осциллятор Ван дер Поля, ≈6 мин.  python -m demos.vdp  [--quick]
from demos.common import run_model

run_model('vdp', repeats=20, trials=20, alphas=41)
