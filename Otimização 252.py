import pandas as pd
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
import numpy as np
from scipy.optimize import Bounds, minimize

# --- Leitura dos dados ---
carteira_ibov = pd.read_csv('Dados/IBOVDia_10-04-25.csv', sep=';', decimal=',', encoding='latin-1')
carteira_ibov.sort_values(by='Código', inplace=True)
ativos_ibov = carteira_ibov['Código'].values
ativos_ibov_SA = [i + '.SA' for i in ativos_ibov]

# --- Definindo datas ---
hoje = date.today()
dois_anos_atras = hoje - relativedelta(years=2)

# --- Baixando os preços ajustados ---
close_adj = yf.download(ativos_ibov_SA, start=dois_anos_atras, end=hoje, auto_adjust=False)['Adj Close']

# --- Cálculo dos retornos logarítmicos diários ---
close_adj_retornos = np.log(close_adj / close_adj.shift(1)).drop(index=close_adj.index[0])
close_adj_retornos.dropna(axis=1, inplace=True)

[TAM, Qtd_ativos] = close_adj_retornos.shape

# --- Parâmetros para a otimização ---
Qtd_pontos_fronteira = 400
pontos_front = np.linspace(0, 1, Qtd_pontos_fronteira)

# --- Cálculo da matriz de covariância e dos retornos médios diários ---
cov_matrix = close_adj_retornos.cov()      # matriz de covariância diária
media_retornos = close_adj_retornos.mean().values  # retornos diários médios

# --- Annualização ---
N = 252  # dias úteis no ano
annual_media = media_retornos * N              # retorno anual
annual_cov = cov_matrix * N                      # matriz de covariância anual

# Função para verificar se a matriz é definida positiva (opcional)
def is_pos_def(A):
    M = np.matrix(A)
    return np.all(np.linalg.eigvals(M) > 0)

is_pos_def(annual_cov)

# --- Restrições da otimização ---
bnds = Bounds(np.zeros(Qtd_ativos), np.ones(Qtd_ativos))
def h(x):
    return sum(x) - 1
cons = [{'type': 'eq', 'fun': h}]
x0 = (1 / Qtd_ativos) * np.ones(Qtd_ativos)

# --- Funções de retorno e risco (em base anual) ---
def calcula_retorno(x):
    return annual_media.dot(x)

def calcula_risco(x):
    # Retorna a volatilidade anual: sqrt(xᵀ * annual_cov * x)
    return np.sqrt((x.dot(annual_cov.values)).dot(x))

# --- Função de otimização para um ponto da fronteira ---
def realiza_otimizacao(aux_ponto_fronteira):
    def funcao_objetivo(x, ponto_fronteira=aux_ponto_fronteira):
        retorno = calcula_retorno(x)
        risco = calcula_risco(x)
        # Balanceia risco e retorno usando o ponto da fronteira (escala de 0 a 1)
        return ponto_fronteira * risco - (1 - ponto_fronteira) * retorno
    xmin = minimize(funcao_objetivo, x0, bounds=bnds, constraints=cons)
    return xmin

# --- Construção da fronteira eficiente ---
def constroi_fronteira():
    carteira_front = []
    retorno = np.zeros(Qtd_pontos_fronteira)
    risco = np.zeros(Qtd_pontos_fronteira)
    pos = 0
    for ponto in pontos_front:
        carteira = realiza_otimizacao(ponto)
        x_opt = carteira.x  
        carteira_front.append(x_opt)
        retorno[pos] = calcula_retorno(x_opt)
        risco[pos] = calcula_risco(x_opt)
        pos += 1
    return retorno, risco, carteira_front

retorno, risco, carteira_front = constroi_fronteira()

# --- Criação do DataFrame para o plot ---
df_front = pd.DataFrame({
    'Retorno': retorno,
    'Risco': risco
})

# Função para formatar os pesos para exibição no hover
def format_weights(pesos):
    return ', '.join([f"{w:.4f}" for w in pesos])

df_front['Pesos'] = [format_weights(p) for p in carteira_front]

# Identifica os portfólios de mínima variância e de máximo retorno
idx_min_variancia = np.argmin(risco)
idx_max_retorno = np.argmax(retorno)
df_front['Tipo'] = 'Normal'
df_front.loc[idx_min_variancia, 'Tipo'] = 'Min Variância'
df_front.loc[idx_max_retorno, 'Tipo'] = 'Max Retorno'

# --- Plot interativo aprimorado com Plotly ---
import plotly.express as px
import plotly.graph_objects as go

# Acrescenta um parâmetro contínuo para a cor (usado aqui apenas para uma escala visual)
df_front['Lambda'] = np.linspace(0, 1, len(df_front))

fig = px.scatter(
    df_front,
    x='Risco',
    y='Retorno',
    color='Lambda',
    color_continuous_scale='Viridis',
    size_max=12,
    hover_data={'Lambda': ':.2f', 'Pesos': True},
    title='Fronteira Eficiente – Portfólios Ótimos (Base Anual)'
)

fig.update_traces(
    marker=dict(
        line=dict(width=1, color='DarkSlateGrey'),
        sizemode='diameter'
    ),
    selector=dict(mode='markers')
)

min_idx = df_front['Tipo'] == 'Min Variância'
max_idx = df_front['Tipo'] == 'Max Retorno'

fig.add_trace(go.Scatter(
    x=df_front.loc[min_idx, 'Risco'],
    y=df_front.loc[min_idx, 'Retorno'],
    mode='markers+text',
    marker=dict(symbol='diamond', size=14),
    text=['Min Var'],
    textposition='top center',
    name='Mínima Variância'
))

fig.add_trace(go.Scatter(
    x=df_front.loc[max_idx, 'Risco'],
    y=df_front.loc[max_idx, 'Retorno'],
    mode='markers+text',
    marker=dict(symbol='star', size=16),
    text=['Max Ret'],
    textposition='bottom right',
    name='Máximo Retorno'
))

fig.update_layout(
    template='plotly_white',
    xaxis=dict(
        title='Risco (Volatilidade Anual)',
        gridcolor='LightGrey',
        gridwidth=0.5,
        griddash='dot'
    ),
    yaxis=dict(
        title='Retorno Anual Esperado',
        gridcolor='LightGrey',
        gridwidth=0.5,
        griddash='dot'
    ),
    coloraxis_colorbar=dict(
        title='Lambda',
        tickformat='.2f'
    ),
    title_font_size=20,
    legend=dict(
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    )
)

fig.show()
