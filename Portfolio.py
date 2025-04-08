# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# Portfolio built: in this case i am selecting manually the period range and the composing of portfolio
# Using de yfinance for the downloads the data
start_date = '2020-04-02'

end_date = '2025-04-02'

ativos = ['TCSA3.SA', 'LWSA3.SA', 'ENJU3.SA', 'BLAU3.SA']

pesos = np.array([0.085, 0.654, 0.0934, 0.1442])

cotacoes = yf.download(ativos, start=start_date, end=end_date, auto_adjust=False)['Adj Close']

cotacoes

# Calculated log returns

cotacoes_log_retorno = np.log(cotacoes/cotacoes.shift(1)).dropna()

cotacoes_log_retorno

# accumulated renturs
retorno_acumulado = (1+cotacoes_log_retorno).cumprod()

retorno_acumulado


# Portfolio 

variação_carteira = (cotacoes_log_retorno*pesos).sum(axis=1)

variação_carteira

# Historical VaR

Var_hist_95 = np.percentile(variação_carteira, 1)

Var_hist_95*100

plt.figure(figsize=(10, 6))
plt.hist(variação_carteira, bins=30, color='skyblue', edgecolor='black')
plt.title("Histograma das Variações Diárias da Carteira")
plt.xlabel("Variação Diária")
plt.ylabel("Frequência")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# Portfolio volatility
cov_matrix = cotacoes_log_retorno.cov()

vol_carteira = np.sqrt(np.dot(pesos.T, np.dot(cov_matrix,pesos)))

vol_carteira

vol_carteira_y = vol_carteira*np.sqrt(252)

vol_carteira_y



