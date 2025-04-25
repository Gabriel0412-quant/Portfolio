import yfinance as yf

from pypfopt import risk_models

from pypfopt import expected_returns

from pypfopt import efficient_frontier

from pypfopt import EfficientFrontier

import numpy as np

import pandas as pd


HK_992 = yf.download('0992.HK', start='2023-02-27', end='2025-02-27')['Close']

HK_992_retornos = np.log(HK_992/HK_992.shift(1)).dropna()

HK_vol_std = HK_992_retornos.std()*np.sqrt(251)

HK_vol_std

HK_vol_ewma = risk_models.exp_cov(HK_992, span= 18)

HK_vol_ewma


import numpy as np
import pandas as pd

def ewma_volatility(price_series, span=20, 
                    lambda_=0.94, 
                    annualization=252, 
                    init_var=None):
 
    # 1) Calcula log-retornos (simples ou log)
    returns = np.log(price_series / price_series.shift(1)).dropna()

    # 2) Define λ a partir de span se não for fornecido
    if lambda_ is None:
        alpha = 2.0 / (span + 1.0)   # α = 2/(span+1) :contentReference[oaicite:7]{index=7}
        lambda_ = 1.0 - alpha       # λ = 1 - α :contentReference[oaicite:8]{index=8}

    # 3) Inicializa σ²
    if init_var is None:
        init_var = returns.iloc[0]**2

    sigma2 = np.empty(len(returns))
    sigma2[0] = init_var

    # 4) Recursão EWMA
    for t in range(1, len(returns)):
        sigma2[t] = lambda_ * sigma2[t-1] + (1 - lambda_) * returns.iloc[t]**2

    # 5) Converte em volatilidade e anualiza
    vol = np.sqrt(sigma2 * annualization)

    # 6) Retorna pd.Series com mesmo índice dos retornos
    return pd.Series(vol, index=returns.index)


ewma_volatility(HK_992)


from pypfopt import risk_models

Sigma_ewma = risk_models.exp_cov(
    prices   = HK_992,
    span     = 32,         # para lambda≈0.94
    frequency=252
)

