
# Markov regime switch
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
import matplotlib.pyplot as plt

data = yf.download("HYPE3.SA")
returns = np.log(data.Close / data.Close.shift(1))
range = (data.High - data.Low)
features = pd.concat([returns, range], axis=1).dropna()
features.columns = ["returns", "range"]
 
# Fit the Markov model
model = hmm.GaussianHMM(
    n_components=3,
    covariance_type="full",
    n_iter=1000,
)
model.fit(features)

# The states
states = pd.Series(model.predict(features), index=data.index[1:])
states.name = "state"
states.hist()
plt.show()

# The plot
color_map = {
    0.0: "green",
    1.0: "orange",
    2.0: "red"
}
(
    pd.concat([data.Close, states], axis=1)
    .dropna()
    .set_index("state", append=True)
    .Close
    .unstack("state")
    .plot(color=color_map, figsize=[16, 12])
)
plt.show()


# Exercício com volatilidade
data = yf.download("HYPE3.SA")

# Calculando o desvio padrão dos retornos
returns = np.log(data.Close / data.Close.shift(1))
std_dev = returns.rolling(window=30).std().dropna()  # Usando uma janela móvel de 30 dias para suavizar o desvio padrão

# Preparando os recursos
features = std_dev.values.reshape(-1, 1)

# Configuração do modelo de Markov
model = hmm.GaussianHMM(
    n_components=2,  # Número de estados ocultos
    covariance_type="full",  # Tipo de matriz de covariância
    n_iter=1000,  # Número de iterações do algoritmo de treinamento
)

# Treinando o modelo
model.fit(features)

# Prevendo os estados (alta ou baixa volatilidade)
states = model.predict(features)

# Plotando os níveis de volatilidade com cores correspondentes aos estados previstos
plt.figure(figsize=[16, 6])
for i, state in enumerate(states):
    color = 'green' if state == 1 else 'red'
    plt.plot(std_dev.index[i+1], std_dev.values[i+1], color=color, marker='o')

plt.ylabel('Volatilidade')
plt.xlabel('Data')
plt.title('Mudança de Regime na Volatilidade do Ibovespa')
plt.show()