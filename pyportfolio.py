
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







