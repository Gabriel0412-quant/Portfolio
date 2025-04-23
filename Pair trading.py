import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
from datetime import datetime

# 1) Defina os ativos e o período
ativo1 = 'PETR3.SA'
ativo2 = 'PETR4.SA'
tickers = [ativo1, ativo2]
inicio = '2024-04-01'
fim    = '2025-04-01'

# 2) Baixe os preços ajustados
#    e monte um DataFrame com as colunas renomeadas
ativos = pd.DataFrame()
for tk in tickers:
    df = yf.download(tk, start=inicio, end=fim, auto_adjust=False)[['Adj Close']]
    df.columns = [tk]
    ativos = pd.concat([ativos, df], axis=1)
ativos.index.name = 'Date'

# 3) Plotagem com matplotlib
plt.figure(figsize=(12, 6))
plt.plot(ativos.index, ativos[ativo1], label=ativo1, linewidth=2)
plt.plot(ativos.index, ativos[ativo2], label=ativo2, linewidth=2)

plt.title('Séries de Preço Ajustado', fontsize=14)
plt.xlabel('Data', fontsize=12)
plt.ylabel('Preço Ajustado (R$)', fontsize=12)
plt.legend()
plt.grid(linestyle='--', alpha=0.5)

# Formata eixo de datas para ficar mais legível
plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.show()

ativos.dropna(inplace=True)

score, pvalue, _ = coint(ativos[ativo1], ativos[ativo2])
pvalue


x1 = ativos[ativo1]

x2 = ativos[ativo2]

x1 = sm.add_constant(x1)

resultado = sm.OLS(x2, x1).fit()

x1 = x1[ativo1]

beta = resultado.params[ativo1]

beta

spread = x2 - beta * x1

spread


# Plotando o spread
plt.figure(figsize=(12, 6))
plt.plot(spread.index, spread, label='Spread', color='blue')
plt.axhline(spread.mean(), color='red', linestyle='--', label='Média do Spread')
plt.title('Spread entre PETR3 e PETR4')
plt.xlabel('Data')
plt.ylabel('Spread')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



teste = adfuller(spread)
teste[1]


z_score = (spread - spread.mean())/ np.std(spread)

z_score


plt.figure(figsize=(12, 6))
plt.plot(z_score.index, z_score, label='z_score', color='blue')
plt.axhline(z_score.mean(), color='red', linestyle='--', label='Média do z_score')
plt.title('z_score entre PETR3 e PETR4')
plt.xlabel('Data')
plt.ylabel('z_score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()






import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
from datetime import datetime
import yfinance as yf
import pandas as pd

# definição de tickers e datas
tickers = ['PETR3.SA', 'PETR44.SA']
inicio  = '2024-04-01'
fim     = '2025-04-020'

ativos_open = pd.DataFrame()

for ticker in tickers:
    # pega série de abertura como Series
    s = yf.download(ticker, start=inicio, end=fim)['Open']
    # renomeia a série para o próprio ticker
    s.name = ticker
    # concatena
    ativos_open = pd.concat([ativos_open, s], axis=1)

ativos_open.index.name = 'Date'
print(ativos_open)


CAIXA = 100000

TAXA = 0

PCT_ORDEM1 = 0.1

PCT_ORDEM2 = 0.1

BANDA_SUPERIOR = 1

BANDA_INFERIOR = -1

vbt_sinal_short = (z_score > BANDA_SUPERIOR).rename("Sinal short" )

vbt_sinal_long = (z_score > BANDA_INFERIOR).rename("Sinal long" )

vbt_sinal_short, vbt_sinal_long = pd.Series.vbt.signal.clean(vbt_sinal_short, vbt_sinal_long, entry_first=False, broadcast_kwargs=dict(colums_from='keep'))

tickers_coluna = pd.Index([ativo1, ativo2], name='tickers')
vbt_ordem = pd.DataFrame(index=ativos.index, columns=tickers_coluna) #entendi
vbt_ordem[ativo1] = np.nan
vbt_ordem[ativo2] = np.nan

vbt_ordem.loc[vbt_sinal_short, ativo1] = -PCT_ORDEM1
vbt_ordem.loc[vbt_sinal_long, ativo1] = PCT_ORDEM1
vbt_ordem.loc[vbt_sinal_short, ativo2] = PCT_ORDEM2
vbt_ordem.loc[vbt_sinal_long, ativo2] = -PCT_ORDEM2

vbt_ordem = vbt_ordem.vbt.fshift(1)

print(vbt_ordem[~vbt_ordem.isnull().any(axis=1)])


