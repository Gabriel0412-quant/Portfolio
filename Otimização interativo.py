import pandas as pd
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
import numpy as np
from scipy.optimize import Bounds, minimize

# Dash & Plotly
from dash import Dash, dcc, html, Input, Output  # :contentReference[oaicite:0]{index=0}
import plotly.express as px
import plotly.graph_objects as go

# --- 1) Preparação dos dados e otimização (base anual) ---
carteira_ibov = pd.read_csv('Dados/IBOVDia_10-04-25.csv', sep=';', decimal=',', encoding='latin-1')
carteira_ibov.sort_values(by='Código', inplace=True)
ativos = [f"{c}.SA" for c in carteira_ibov['Código'].values]

hoje = date.today()
dois_anos = hoje - relativedelta(years=2)
df_close = yf.download(ativos, start=dois_anos, end=hoje, auto_adjust=False)['Adj Close']
ret = np.log(df_close / df_close.shift(1)).dropna().dropna(axis=1)

# Parâmetros
N = 252  # dias úteis/ano
annual_mu  = ret.mean().values * N
annual_cov = ret.cov().values * N
n = len(ativos)

bounds = Bounds(np.zeros(n), np.ones(n))
cons = ({'type': 'eq', 'fun': lambda w: np.sum(w)-1},)
x0 = np.ones(n)/n

def port_return(w): return annual_mu.dot(w)
def port_risk(w):   return np.sqrt(w @ annual_cov @ w)

# Gera fronteira eficiente
m = 200
lams = np.linspace(0, 1, m)
weights = np.array([minimize(lambda w, λ=λ: λ*port_risk(w)-(1-λ)*port_return(w),
                             x0, bounds=bounds, constraints=cons).x
                    for λ in lams])
rets  = np.array([port_return(w) for w in weights])
risks = np.array([port_risk(w)   for w in weights])

# DataFrame para plot
df = pd.DataFrame({
    'Risco': risks,
    'Retorno': rets,
    'Lambda': lams
})
df['Pesos'] = [', '.join(f"{w:.3f}" for w in ws) for ws in weights]
df['Tipo'] = 'Normal'
df.loc[df['Risco'].idxmin(), 'Tipo'] = 'Min Variância'
df.loc[df['Retorno'].idxmax(), 'Tipo'] = 'Max Retorno'

# --- 2) Montagem do Dash App ---
app = Dash(__name__)

app.layout = html.Div([
    html.H2("Fronteira Eficiente – Composição Dinâmica"),
    dcc.Graph(id='front-scatter', config={'displayModeBar': False}),
    html.H4("Composição do Portfólio Selecionado:"),
    dcc.Graph(id='weight-bar', config={'displayModeBar': False})
], style={'width':'80%','margin':'auto'})

# Callback para atualizar o scatter
@app.callback(
    Output('front-scatter', 'figure'),
    Input('front-scatter', 'clickData')
)
def update_scatter(clickData):
    fig = px.scatter(df, x='Risco', y='Retorno',
                     color='Tipo',
                     color_discrete_map={'Normal':'lightgrey',
                                         'Min Variância':'blue',
                                         'Max Retorno':'red'},
                     hover_data=['Pesos'],
                     title="Clique em um ponto para ver os pesos")  # :contentReference[oaicite:1]{index=1}
    if clickData:
        pt = clickData['points'][0]
        fig.add_trace(go.Scatter(
            x=[pt['x']], y=[pt['y']],
            mode='markers', marker=dict(size=15, symbol='x'),
            name='Selecionado'
        ))
    fig.update_layout(template='plotly_white')
    return fig

# Callback para atualizar o bar chart de pesos
@app.callback(
    Output('weight-bar', 'figure'),
    Input('front-scatter', 'clickData')
)
def update_bar(clickData):
    if clickData:
        idx = clickData['points'][0]['pointIndex']
    else:
        idx = int(np.argmin(risks))  # default: mínima variância
    w = weights[idx]
    fig = go.Figure([go.Bar(x=ativos, y=w)])
    fig.update_layout(
        title=f"Pesos do Portfólio (λ={lams[idx]:.2f})",
        xaxis_tickangle=-45,
        yaxis_title="Peso",
        template='plotly_white'
    )
    return fig

if __name__ == '__main__':
    app.run(debug=True)
