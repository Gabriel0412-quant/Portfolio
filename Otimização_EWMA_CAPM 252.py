import pandas as pd
import numpy as np
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
from scipy.optimize import Bounds, minimize
import plotly.express as px
import plotly.graph_objects as go

# --- 1) Parâmetros CAPM e EWMA ---
R_f    = 0.14    # taxa livre de risco anual (14%)
prêmio = 0.06    # prêmio de risco de mercado (6%)
λ_ewma = 0.94    # decay factor EWMA
N      = 252     # dias úteis no ano

# --- 2) Baixa dados de ativos e benchmark (IBOV) ---
carteira = pd.read_csv(
    'Dados/IBOVDia_10-04-25.csv',
    sep=';', decimal=',', encoding='latin-1'
)
ativos = [f"{c}.SA" for c in carteira['Código'].sort_values().astype(str)]

hoje     = date.today()
dois_anos = hoje - relativedelta(years=2)

df_close = yf.download(
    ativos + ['^BVSP'],
    start=dois_anos,
    end=hoje,
    auto_adjust=False
)['Adj Close']

df_ret = np.log(df_close / df_close.shift(1)).dropna()

# --- 3) Separa retornos ---
ret_ativos = df_ret[ativos]
ret_ibov   = df_ret['^BVSP']

ret_ativos

# --- 4) Calcula betas pelo CAPM ---
betas = ret_ativos.apply(lambda s: s.cov(ret_ibov) / ret_ibov.var())
# Vetor de retornos esperados via CAPM (já anual)
mu_capm = R_f + betas.values * prêmio

mu_capm

# --- 5) Estima covariância via EWMA ---
# Inicializa S com covariância amostral diária
S = ret_ativos.cov().values
for t in range(1, len(ret_ativos)):
    r = ret_ativos.iloc[t].values.reshape(-1,1)
    S = λ_ewma * S + (1 - λ_ewma) * (r @ r.T)
cov_ewma_annual = S * N


# --- 6) Otimização Markowitz (CAPM + EWMA) ---
n = len(ativos)
bounds = Bounds(np.zeros(n), np.ones(n))
cons   = ({'type':'eq', 'fun': lambda w: np.sum(w) - 1},)
x0     = np.ones(n) / n

def port_return(w):
    return mu_capm.dot(w)

def port_vol(w):
    return np.sqrt(w @ cov_ewma_annual @ w)

def solve_point(alpha):
    def obj(w):
        return alpha * port_vol(w) - (1 - alpha) * port_return(w)
    res = minimize(obj, x0, bounds=bounds, constraints=cons)
    return res.x

m      = 400
alphas = np.linspace(0, 1, m)
weights = np.array([solve_point(a) for a in alphas])
rets    = np.array([port_return(w) for w in weights])
vols    = np.array([port_vol(w)    for w in weights])


# --- 7) Prepara DataFrame para plot ---
df = pd.DataFrame({
    'Retorno Anual (%)':     rets * 100,
    'Volatilidade Anual (%)': vols * 100,
    'Alpha':                  alphas
})
df['Pesos'] = [', '.join(f"{w:.2%}" for w in ws) for ws in weights]
df['Tipo']   = 'Normal'
df.loc[vols.argmin(), 'Tipo'] = 'Mínima Vol.'
df.loc[rets.argmax(), 'Tipo'] = 'Máx Ret.'

# --- 8) Plot interativo ---
import matplotlib.pyplot as plt
import numpy as np

# Supondo que você já tenha:
# df: DataFrame com colunas 'Volatilidade Anual (%)', 'Retorno Anual (%)', 'Tipo' e 'Pesos'
# vols, rets: arrays com volatilidades e retornos
# df['Tipo'] categorizado em 'Normal', 'Mínima Vol.', 'Máx Ret.'

# 1) Mapear cores
color_map = {
    'Normal': 'lightgrey',
    'Mínima Vol.': 'blue',
    'Máx Ret.': 'red'
}
colors = df['Tipo'].map(color_map)

# 2) Cria figura
fig, ax = plt.subplots(figsize=(10, 6))

# 3) Scatter principal
for tipo in df['Tipo'].unique():
    mask = df['Tipo'] == tipo
    ax.scatter(
        df.loc[mask, 'Volatilidade Anual (%)'],
        df.loc[mask, 'Retorno Anual (%)'],
        label=tipo,
        c=color_map[tipo],
        edgecolor='k',
        linewidth=0.8,
        s=60,
        alpha=0.8
    )

# 4) Anotações dos pontos especiais
idx_min = np.argmin(vols)
idx_max = np.argmax(rets)

# Mínima volatilidade
ax.scatter(
    vols[idx_min]*100,
    rets[idx_min]*100,
    marker='D', s=100, c='blue', edgecolor='k', zorder=5
)
ax.text(
    vols[idx_min]*100, rets[idx_min]*100 + 0.5,
    'Min Vol', ha='center', va='bottom', fontsize=10, fontweight='bold'
)

# Máximo retorno
ax.scatter(
    vols[idx_max]*100,
    rets[idx_max]*100,
    marker='*', s=150, c='red', edgecolor='k', zorder=5
)
ax.text(
    vols[idx_max]*100 + 0.5, rets[idx_max]*100,
    'Max Ret', ha='left', va='center', fontsize=10, fontweight='bold'
)

# 5) Grid pontilhado
ax.grid(True, linestyle=':', color='grey', alpha=0.6)

# 6) Labels e título
ax.set_xlabel('Volatilidade Anual (%)')
ax.set_ylabel('Retorno Anual (%)')
ax.set_title('Fronteira Eficiente – CAPM + EWMA')

# 7) Legenda
ax.legend(title='Tipo de Portfólio', loc='upper right')

plt.tight_layout()
plt.show()

