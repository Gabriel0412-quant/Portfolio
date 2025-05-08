import yfinance as yf
from binance.client import Client
import pandas as pd

Client = Client()

# Fetch BTC/BRL data
btcbrl = Client.get_klines(symbol="BTCBRL", interval=Client.KLINE_INTERVAL_1DAY)

# Convert data to float and clean up unnecessary columns
btcbrl = [list(map(float, line)) for line in btcbrl]
for line in btcbrl:
    del line[5:]

# Create a DataFrame and process the data
btcbrl = pd.DataFrame(btcbrl, columns=["Date", "Open", "High", "Low", "Close"])
btcbrl.set_index("Date", inplace=True)
btcbrl.index = pd.to_datetime(btcbrl.index, unit="ms")
btcbrl['Close'] = pd.to_numeric(btcbrl['Close'])

# Limitar os dados para teste (opcional)
btcbrl = btcbrl.tail(50)  # Use apenas as últimas 100 linhas para o gráfico

# Display the processed DataFrame


# Fetch ETH/BRL historical data
ethbrl = Client.get_historical_klines("ETHBRL", Client.KLINE_INTERVAL_1DAY, "1 year ago UTC")

ethbrl = [list(map(float, line)) for line in ethbrl]
for line in ethbrl:
    del line[5:]
    
ethbrl = pd.DataFrame(ethbrl, columns=["Date", "Open", "High", "Low", "Close"])
ethbrl.set_index("Date", inplace=True)
ethbrl.index = pd.to_datetime(ethbrl.index, unit="ms")
ethbrl['Close'] = pd.to_numeric(ethbrl['Close'])

ethbrl = ethbrl.tail(50)  # Use apenas as últimas 100 linhas para o gráfico


import plotly.express as px
import plotly.graph_objects as go

# Plotar grafcos com Plotly de forma simples
fig = px.line(btcbrl, x=btcbrl.index, y=btcbrl['Close'], title='BTC/BRL Price')
fig.show()


# Plotar gráficos com Plotly de forma mais complexa em escala esquisita
fig = go.Figure()
fig.add_trace(go.Scatter(x=btcbrl.index, y=btcbrl['Close'], mode='lines', name='BTC/BRL'))
fig.add_trace(go.Scatter(x=ethbrl.index, y=ethbrl['Close'], mode='lines', name='ETH/BRL'))
fig.update_layout(title='BTC/BRL and ETH/BRL Prices')
fig.show()

# Plotar gráficos com Plotly de forma mais complexa com subplots sendo um grafico em cima do outro
from plotly.subplots import make_subplots

fig = make_subplots(rows=2, cols=1, subplot_titles=('BTC/BRL Price', 'ETH/BRL Price'))
fig.add_trace(go.Scatter(x=btcbrl.index, y=btcbrl['Close'], mode='lines', name='BTC/BRL'), row=1, col=1)
fig.add_trace(go.Scatter(x=ethbrl.index, y=ethbrl['Close'], mode='lines', name='ETH/BRL'), row=2, col=1)
fig.update_layout(title='BTC/BRL and ETH/BRL Prices')
fig.show()


# Plotar gráficos com Plotly de forma mais complexa com subplots sendo os dois dados no mesmo gráfico em escala normal
fig.make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(x=btcbrl.index, y=btcbrl['Close'], mode='lines', name='BTC/BRL'), secondary_y=False)
fig.add_trace(go.Scatter(x=ethbrl.index, y=ethbrl['Close'], mode='lines', name='ETH/BRL'), secondary_y=True)
fig.update_layout(title='BTC/BRL and ETH/BRL Prices with Secondary Y-Axis')
fig.show()


# Criando grafico de dispersão

retornos_btc = pd.DataFrame()
retornos_eth = pd.DataFrame()

retornos_btc['Close_BTC'] = btcbrl['Close'].pct_change()
retornos_eth['Close_ETH'] = ethbrl['Close'].pct_change()

# Grafico de dispersão com Plotly Express
fig = px.scatter(x=retornos_btc['Close_BTC'], y=retornos_eth['Close_ETH'], title='BTC vs ETH Returns')
fig.update_traces(marker=dict(size=5, opacity=0.5, line=dict(width=1, color='DarkSlateGrey')))
fig.show()

df_juntos = pd.merge(retornos_btc, retornos_eth, left_index=True, right_index=True, suffixes=('_BTC', '_ETH'), how='inner')

fig = px.scatter(df_juntos, x='Close_BTC', y='Close_ETH', title='BTC vs ETH Returns')
fig.update_traces(marker=dict(size=5, opacity=0.5, line=dict(width=1, color='DarkSlateGrey')))
fig.show()

# Grafico histograma com Plotly Express

fig = px.histogram(df_juntos, x='Close_BTC')
fig.show()

# Grafico histograma com Plotly Graph Objects
fig = go.Figure()
fig.add_trace(go.Histogram(x=df_juntos['Close_BTC'], name='BTC Returns', opacity=0.75))
fig.show()

# Box plot com Plotly Express
fig = px.box(df_juntos, y='Close_BTC', title='BTC Returns Box Plot')
fig.show()

# Grafico de pizza com Plotly Express
tickers = ['VALE3.SA', 'ALOS3.SA', 'MRVE3.SA']

pesos = [0.4, 0.3, 0.3]

fig = px.pie(pesos, values=pesos, names=tickers, title='Portfolio Allocation')
fig.show()

import investpy as ivp

tickers = ['VALE3.SA', 'ALOS3.SA', 'MRVE3.SA']

pesos = [0.4, 0.3, 0.3]

marketcap_tickers = pd.DataFrame()

for i in tickers:
    df = ivp.get_stock_information(i, country='brazil')['Market Cap']
    df.rename(i, inplace=True)
    df.columns = [i]  # Fixed typo: 'collums' -> 'columns'
    marketcap_tickers = pd.concat([marketcap_tickers, df], axis=1)


marketcap_tickers.columns = tickers

marketcap_tickers


import investpy as ivp
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

tickers = ['VALE3.SA', 'ALOS3.SA', 'MRVE3.SA']
pesos = [0.4, 0.3, 0.3]
marketcap_tickers = pd.DataFrame()

for i in tickers:
    info = ivp.get_stock_information(i, country='brazil')
    market_cap = info.loc['Market Cap']
    df = pd.DataFrame({i: [market_cap]})
    marketcap_tickers = pd.concat([marketcap_tickers, df], axis=1)

print(marketcap_tickers)


fig = go.Figure(go.bar(x=marketcap_tickers.iloc[-1], y=marketcap_tickers.columns[0], orientation = 'h'))
fig.update_layout(title='Market Cap of Stocks')
fig.show()












