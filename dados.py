import streamlit as st
import pandas as pd
import yfinance as yf

st.title("Análise de Ações com Streamlit")


tickers = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'ABEV3.SA', 'MGLU3.SA']

precos = yf.download(tickers, start="2023-01-01", end="2023-10-01", auto_adjust=False)['Adj Close']

st.subheader("Tabela de Preços Ajustados")
st.dataframe(precos, width=1000, height=500)

st.subheader('Função table')
st.table(precos, width=1000, height=500)

st.dataframe(precos.head(10).style_highlight_max(axis=0), width=1000, height=500)

st.metric('Temperatura', value='20ºC', delta='1ºC')
