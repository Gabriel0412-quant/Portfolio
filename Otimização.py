# Fazer modelo de otimização de markovitz com calculo de CAPM como retorno e risco como vol EWMZ

import pandas as pd
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
import numpy as np

carteira_ibov = pd.read_csv('Dados\IBOVDia_10-04-25.csv', sep=';', decimal=',', encoding='latin-1')

carteira_ibov.sort_values(by='Código', inplace=True)

ativos_ibov = carteira_ibov['Código'].values

ativos_ibov

ativos_ibov_SA =  [i + '.SA' for i in ativos_ibov]

ativos_ibov_SA

hoje = date.today()

hoje


hoje = date.today()

dois_anos_atras = hoje - relativedelta(years=2)

close_adj = yf.download(ativos_ibov_SA, start=dois_anos_atras, end=hoje, auto_adjust=False)['Adj Close']

close_adj_retornos = np.log(close_adj/close_adj.shift(1)).drop(index=close_adj.index[0])

close_adj_retornos