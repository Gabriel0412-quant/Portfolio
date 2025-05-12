# Study 
import pandas as pd
import numpy as np
import yfinance as yf

ativos = ['ABEV3.SA', 'ITSA4.SA', 'PETR4.SA', 'VALE3.SA']

ativos_passados_start = '2016-04-01'

ativos_passados_end = '2018-12-28'

ativos_futuros_start = '2019-01-02'

ativos_futuros_end = '2020-12-30'


carteira_passado = yf.download(ativos, start=ativos_passados_start, end=ativos_passados_end, auto_adjust=False)['Adj Close']

carteira_futuro = yf.download(ativos, start=ativos_futuros_start, end=ativos_futuros_end, auto_adjust=False)['Adj Close']

carteira_futuro

carteira_passado

pesos = np.array([0.25, 0.25, 0.25, 0.25])

cf_anualizado = (carteira_futuro.iloc[-1]-carteira_futuro.iloc[0])/ carteira_futuro.iloc[0]
cf_anualizado = ((1+cf_anualizado)**(12/24))-1
cf_anualizado

cf_anualizado_carteira = cf_anualizado.dot(pesos)
cf_anualizado_carteira

carteira_futuro_retornos = carteira_futuro.pct_change()
carteira_futuro_retornos

cov_fut = carteira_futuro_retornos.cov()

cov_fut

vol_fut_diaria = np.sqrt(np.dot(pesos.T, np.dot(cov_fut, pesos)))

vol_fut_anual = vol_fut_diaria*np.sqrt(252)

vol_fut_anual

from pypfopt import risk_models

from pypfopt import expected_returns

from pypfopt import efficient_frontier

from pypfopt import EfficientFrontier


selic_aa = 0.1425

selic_diaria = (1+selic_aa)**(1/252)-1


carteira = pd.read_csv('Dados\Analise_Quant_Cenarios_Avancados_carteira.csv')

carteira.index = carteira['Date']

carteira.drop(columns=('Date'), inplace=True)

carteira

cov_matrix = risk_models.CovarianceShrinkage(carteira).ledoit_wolf()

cov_matrix

ibov2 = pd.read_csv('Dados\ibov2.csv')

ibov2.index = ibov2['Date']

ibov2.drop(columns=('Date'), inplace=True)

ibov2

capm_Carteira = expected_returns.capm_return(carteira, market_prices=ibov2, risk_free_rate=selic_diaria)

capm_Carteira


ls = EfficientFrontier(capm_Carteira, cov_matrix, weight_bounds=(None, None))

ls.min_volatility()

ls_pesos = ls.clean_weights()

ls_pesos




