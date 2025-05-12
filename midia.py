import streamlit as st
import pandas as pd
import yfinance as yf

tikers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

precos = yf.download(tikers, start='2023-01-01', end='2024-01-01', auto_adjust=False)['Adj Close']

st.title('Funções de graficos para ações')

st.header('funções nativas')

st.dataframe(precos)

st.subheader('Gráfico de linha')

st.line_chart(precos)

st.area_chart(precos)

st.bar_chart(precos)


st.markdown('---')


st.header('Funções do Matplotlib')

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 5))
ax= plt.axes()
ax.plot(precos.index, precos['AAPL'], label='AAPL')
plt.title('AAPL')

st.pyplot(fig)

st.markdown('---')


st.header('Funções do Plotly')

import plotly.express as px


fig = px.line(precos, title='Açoes')

st.plotly_chart(fig, use_container_width=True)


import plotly.graph_objects as go

fig = go.Figure()
fig.add_trace(go.Scatter(x=precos.index, y=precos['AAPL'], mode='lines', name='AAPL'))
fig.add_trace(go.Scatter(x=precos.index, y=precos['MSFT'], mode='lines', name='MSFT'))
fig.update_layout(title='Açoes', xaxis_title='Data', yaxis_title='Preço')
st.plotly_chart(fig, use_container_width=True)







