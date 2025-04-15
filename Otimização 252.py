# Fazer modelo de otimização de markovitz com calculo de CAPM como retorno e risco como vol EWMZ

import pandas as pd
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta
import numpy as np
from scipy.optimize import Bounds
from scipy.optimize import minimize

carteira_ibov = pd.read_csv('Dados\IBOVDia_10-04-25.csv', sep=';', decimal=',', encoding='latin-1')

carteira_ibov.sort_values(by='Código', inplace=True)

ativos_ibov = carteira_ibov['Código'].values

ativos_ibov

ativos_ibov_SA =  [i + '.SA' for i in ativos_ibov]

ativos_ibov_SA

hoje = date.today()

hoje

dois_anos_atras = hoje - relativedelta(years=2)

close_adj = yf.download(ativos_ibov_SA, start=dois_anos_atras, end=hoje, auto_adjust=False)['Adj Close']

close_adj_retornos = np.log(close_adj/close_adj.shift(1)).drop(index=close_adj.index[0])

close_adj_retornos.dropna(axis=1, inplace=True)

close_adj_retornos

[TAM, Qtd_ativos] = close_adj_retornos.shape

TAM

Qtd_ativos

Qtd_pontos_fronteira = 400

pontos_front = np.linspace(0, 1, Qtd_pontos_fronteira)

cov_matrix = close_adj_retornos.cov()

cov_matrix

media_retornos = close_adj_retornos.mean().values

def is_pos_def(A):
    M = np.matrix(A)
    return np.all(np.linalg.eigvals(M)>0)

is_pos_def(cov_matrix)

bnds = Bounds(np.zeros(Qtd_ativos), np.ones(Qtd_ativos))

def h(x):
    return sum(x)-1

cons = [{'type': 'eq', 'fun': h}]


x0 = (1/Qtd_ativos)*np.ones(Qtd_ativos)

sum(x0)

def calcula_retorno(x):
    return media_retornos.dot(x)

def calcula_risco(x):
    return (x.dot(cov_matrix.values)).dot(x)

def realiza_otimizacao(aux_ponto_fronteira):
    def funcao_objetivo(x, ponto_fronteira=aux_ponto_fronteira):
        retorno = calcula_retorno(x)
        risco = calcula_risco(x)
        # Use o ponto da fronteira (um escalar) em vez de todo o array pontos_front
        return ponto_fronteira * risco - (1 - ponto_fronteira) * retorno
    
    xmin = minimize(funcao_objetivo, x0, bounds=bnds, constraints=cons)
    return xmin


def constroi_fronteira():
    carteira_front = []
    retorno = np.zeros(Qtd_pontos_fronteira)
    risco = np.zeros(Qtd_pontos_fronteira)
    pos = 0
    for pontos in pontos_front:
        # Realiza a otimização e obtém o resultado (OptimizeResult)
        carteira = realiza_otimizacao(pontos)
        # Extrai o vetor de pesos ótimo
        x_opt = carteira.x  
        carteira_front.append(x_opt)  # ou mantenha o OptimizeResult conforme sua escolha
        
        retorno[pos] = calcula_retorno(x_opt)
        risco[pos] = calcula_risco(x_opt)
        pos += 1
    return retorno, risco, carteira_front


retorno, risco, carteira_front = constroi_fronteira()

retorno
risco
carteira_front



import pandas as pd
import numpy as np
import plotly.express as px

# Supondo que essas variáveis já existam após sua otimização:
# retorno: array com retornos de cada ponto da fronteira
# risco: array com riscos correspondentes
# carteira_front: lista de arrays de pesos para cada carteira

# Crie um DataFrame com os dados
df_front = pd.DataFrame({
    'Retorno': retorno,
    'Risco': risco
})

# Formate os pesos como string para exibição no hover
def format_weights(pesos):
    return ', '.join([f"{w:.4f}" for w in pesos])

df_front['Pesos'] = [format_weights(p) for p in carteira_front]

# Identificar o portfólio de mínima variância e de máximo retorno:
idx_min_variancia = np.argmin(risco)
idx_max_retorno = np.argmax(retorno)

# Crie uma coluna para identificar o tipo do portfólio:
# Todos serão inicialmente "Normal"
df_front['Tipo'] = 'Normal'
df_front.loc[idx_min_variancia, 'Tipo'] = 'Min Variância'
df_front.loc[idx_max_retorno, 'Tipo'] = 'Max Retorno'

# Crie o gráfico interativo com Plotly Express
fig = px.scatter(
    df_front,
    x='Risco',
    y='Retorno',
    color='Tipo',              # Diferencia os pontos por tipo
    hover_data=['Pesos'],      # Exibe os pesos no hover
    title='Fronteira Eficiente - Portfólios Ótimos'
)

# Personalize o layout (opcional)
fig.update_layout(
    template='simple_white',
    xaxis_title='Risco (Volatilidade)',
    yaxis_title='Retorno Esperado'
)

fig.show()


def melhor_portfolio_por_risco(risk_target, risco, retorno, carteira_front):
    """
    Dado um risco máximo permitido (risk_target) e os arrays de risco, retorno e carteiras,
    retorna o vetor de pesos, o retorno e o risco do portfólio que tem o maior retorno 
    dentro do risco permitido.
    """
    # Encontre os índices dos portfólios que têm risco <= risk_target
    indices_validos = np.where(risco <= risk_target)[0]
    
    if len(indices_validos) == 0:
        # Nenhum portfólio atende à restrição de risco
        return None, None, None
    
    # Entre os portfólios válidos, escolha aquele com o maior retorno
    idx_otimo = indices_validos[np.argmax(retorno[indices_validos])]
    
    return carteira_front[idx_otimo], retorno[idx_otimo], risco[idx_otimo]


# Suponha que esses arrays já foram gerados:
retorno, risco, carteira_front = constroi_fronteira()

# Defina o risco-alvo
risk_target = 0.15  # Por exemplo, 15% de volatilidade

# Obtenha o portfólio ótimo de acordo com esse risco
pesos_otimos, ret_otimo, risco_otimo = melhor_portfolio_por_risco(risk_target, risco, retorno, carteira_front)

if pesos_otimos is not None:
    print("Pesos Ótimos:", pesos_otimos)
    print("Retorno:", ret_otimo)
    print("Risco:", risco_otimo)
else:
    print("Nenhum portfólio atende ao risco alvo.")

