import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

# Downloading PETR4 stock data
petr4 = yf.download('PETR4.SA', start='2021-08-30', end='2023-08-30', auto_adjust=False)['Adj Close']

petr4_retornos = petr4.pct_change().dropna()

petr4_retornos

petr4
# analise do P-valor

resultado = adfuller(petr4['PETR4.SA'])

resultado[1]

resultado2 = adfuller(petr4_retornos['PETR4.SA'])

resultado2[1]


ativo1 = 'PETR3.SA'

ativo2 = 'PETR4.SA'

tickers = [ativo1, ativo2]

inicio = '2021-01-01'
fim = '2022-08-01'

ativos = pd.DataFrame()

for ticker in tickers:
    # Baixa como DataFrame de uma coluna
    df = yf.download(ticker, start=inicio, end=fim, auto_adjust=False)[['Adj Close']]
    # Renomeia a coluna para o próprio ticker
    df.columns = [ticker]
    # Concatena à matriz final
    ativos = pd.concat([ativos, df], axis=1)

ativos.index.name = 'Date'

ativos

fig = go.Figure()
fig.add_trace(go.Scatter(x=ativos.index, y=ativos[ativo1], name= ativo1))
fig.add_trace(go.Scatter(x=ativos.index, y=ativos[ativo2], name= ativo2))
fig.update_layout(title_text= 'Preços', template= 'simple_white')
# depois de montar o fig...
fig.show(renderer="browser")



