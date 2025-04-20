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

# Calculated log returns and clearing de dataframe

cotacoes_log_retorno = np.log(cotacoes/cotacoes.shift(1)).dropna()

cotacoes_log_retorno

# accumulated renturs
retorno_acumulado = (1+cotacoes_log_retorno).cumprod()

retorno_acumulado


# Adding the columns for the built de total renturn, result in portfolio day variation

variação_carteira = (cotacoes_log_retorno*pesos).sum(axis=1)

variação_carteira

retorno_acumulado_carteira = (retorno_acumulado*pesos).sum(axis=1)

retorno_acumulado_carteira

# Historical VaR and plot histogram graph

Var_hist_95 = np.percentile(variação_carteira, 5)

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

# Calculate beta portfolio

benchmark = yf.download("^BVSP", start=start_date, end=end_date)['Close']

benchmark_log_renturs = np.log(benchmark/benchmark.shift(1)).dropna()

benchmark_log_renturs

uniao = pd.concat([benchmark_log_renturs, variação_carteira], axis=1).dropna()

uniao.columns = ['IBOV', 'Carteira']

uniao

import statsmodels.api as sm

y = uniao['Carteira']
x = uniao['IBOV']

x = sm.add_constant(x)

modelo = sm.OLS(y, x)

resultado = modelo.fit()

resultado.params[0]

resultado.params[1]

beta_carteira = resultado.params[1]

beta_carteira

beta_cov = np.cov(uniao['Carteira'], uniao['IBOV'])[0, 1]

beta_cov

# Variância do mercado
var_mercado = np.var(uniao['IBOV'], ddof=1)  # ddof=1 para amostra

# Beta
beta = beta_cov / var_mercado
print(f'Beta da carteira: {beta:.4f}')

# Parametrical VaR

from scipy.stats import norm

media = np.mean(variação_carteira)

desvpad = vol_carteira

para_VaR_95 = norm.ppf(1-0.99, media, desvpad)

para_VaR_95*100

para_VaR_95_y = np.sqrt(21)*para_VaR_95

para_VaR_95_y

# Max Drawndown

pico = retorno_acumulado_carteira.expanding(min_periods=1).max()

pico

dd = (retorno_acumulado_carteira/pico)-1

drawndown = dd.min()

drawndown

