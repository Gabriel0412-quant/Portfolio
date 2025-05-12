import yfinance as yf

from pypfopt import risk_models

from pypfopt import expected_returns

from pypfopt import efficient_frontier

from pypfopt import EfficientFrontier

import numpy as np

import pandas as pd


HK_992 = yf.download('ANIM3.SA', start='2024-02-27', end='2025-02-27', auto_adjust=False)['Close']

HK_992_retornos = np.log(HK_992/HK_992.shift(1)).dropna()

HK_vol_std = HK_992_retornos.std()*np.sqrt(251)

HK_vol_std

HK_vol_ewma = risk_models.exp_cov(HK_992, span= 32)

HK_vol_ewma




import numpy as np
import pandas as pd

def ewma_volatility(price_series, span=20, 
                    lambda_=0.94, 
                    annualization=252, 
                    init_var=None):
 
    # 1) Calcula log-retornos (simples ou log)
    returns = np.log(price_series / price_series.shift(1)).dropna()

    # 2) Define λ a partir de span se não for fornecido
    if lambda_ is None:
        alpha = 2.0 / (span + 1.0)   # α = 2/(span+1) :contentReference[oaicite:7]{index=7}
        lambda_ = 1.0 - alpha       # λ = 1 - α :contentReference[oaicite:8]{index=8}

    # 3) Inicializa σ²
    if init_var is None:
        init_var = returns.iloc[0]**2

    sigma2 = np.empty(len(returns))
    sigma2[0] = init_var

    # 4) Recursão EWMA
    for t in range(1, len(returns)):
        sigma2[t] = lambda_ * sigma2[t-1] + (1 - lambda_) * returns.iloc[t]**2

    # 5) Converte em volatilidade e anualiza
    vol = np.sqrt(sigma2 * annualization)

    # 6) Retorna pd.Series com mesmo índice dos retornos
    return pd.Series(vol, index=returns.index)


ewma_volatility(HK_992)


from pypfopt import risk_models

Sigma_ewma = risk_models.exp_cov(
    prices   = HK_992,
    span     = 32,         # para lambda≈0.94
    frequency=252
)




import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Função para calcular a volatilidade EWMA manualmente
def calcular_ewma_volatilidade(retornos, lambda_valor=0.94, anualizar=True):
    """
    Calcula a volatilidade EWMA com um lambda específico
    
    Parâmetros:
    retornos: Série de retornos logarítmicos diários
    lambda_valor: Fator de decaimento (0.94 é o padrão RiskMetrics)
    anualizar: Se True, anualiza a volatilidade multiplicando por sqrt(252)
    
    Retorna:
    Série pandas com volatilidades EWMA diárias
    """
    # Inicializar a variância como a variância da amostra inicial
    variancia = retornos.iloc[:30].var()
    variancias = [variancia]
    
    # Calcular variâncias EWMA recursivamente
    for i in range(30, len(retornos)):
        # Fórmula EWMA: σ²ₜ = λ·σ²ₜ₋₁ + (1-λ)·r²ₜ₋₁
        variancia = lambda_valor * variancia + (1 - lambda_valor) * retornos.iloc[i-1]**2
        variancias.append(variancia)
    
    # Converter lista para série pandas
    volatilidade = pd.Series(np.sqrt(variancias), index=retornos.index[30:])
    
    # Anualizar se solicitado
    if anualizar:
        volatilidade = volatilidade * np.sqrt(252)
    
    return volatilidade

# Baixar dados do ativo ASAI3.SA
def baixar_e_calcular_volatilidade(ticker, data_inicio, data_fim, lambda_valor=0.94):
    print(f"Baixando dados de {ticker}...")
    dados = yf.download(ticker, start=data_inicio, end=data_fim, progress=False, auto_adjust=False)
    
    # Calcular retornos logarítmicos diários
    precos_ajustados = dados['Adj Close']
    retornos = np.log(precos_ajustados / precos_ajustados.shift(1)).dropna()
    
    print(f"Calculando volatilidade EWMA para {ticker} com lambda={lambda_valor}...")
    vol_ewma = calcular_ewma_volatilidade(retornos, lambda_valor)
    
    # Calcular volatilidade padrão (utilizando desvio padrão móvel de 30 dias) para comparação
    vol_padrao = retornos.rolling(window=30).std() * np.sqrt(252)
    
    # Criar DataFrame com os resultados
    resultados = pd.DataFrame({
        'Preço': precos_ajustados[vol_ewma.index],
        'Retorno': retornos[vol_ewma.index],
        'Volatilidade EWMA': vol_ewma,
        'Volatilidade Padrão (30d)': vol_padrao[vol_ewma.index]
    })
    
    return resultados

# Parâmetros principais
ticker = "ASAI3.SA"  # Assaí
data_inicio = "2023-02-27"
data_fim = "2025-02-27"
lambda_valor = 0.94  # Valor padrão utilizado pelo RiskMetrics

# Executar análise
resultados = baixar_e_calcular_volatilidade(ticker, data_inicio, data_fim, lambda_valor)

# Exibir estatísticas resumidas
print("\nEstatísticas resumidas da volatilidade EWMA anualizada:")
print(resultados['Volatilidade EWMA'].describe())

# Plotar resultados
plt.figure(figsize=(14, 10))

# Gráfico de preços
plt.subplot(3, 1, 1)
plt.plot(resultados['Preço'])
plt.title(f'Preço de {ticker}')
plt.grid(True)

# Gráfico de retornos
plt.subplot(3, 1, 2)
plt.plot(resultados['Retorno'])
plt.title(f'Retornos Logarítmicos Diários de {ticker}')
plt.grid(True)

# Gráfico de Volatilidade
plt.subplot(3, 1, 3)
plt.plot(resultados['Volatilidade EWMA'], label='EWMA (λ=0.94)')
plt.plot(resultados['Volatilidade Padrão (30d)'], label='Desvio Padrão (30d)', alpha=0.7)
plt.title(f'Comparação de Volatilidades Anualizadas de {ticker}')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Para exportar os resultados (opcional)
# resultados.to_excel(f'{ticker}_volatilidade_ewma.xlsx')

print("Volatilidade EWMA mais recente:", resultados['Volatilidade EWMA'].iloc[-1])