# QuantStats is a Python library for performance and risk analytics of financial portfolios.
import quantstats as qs
import yfinance as yf

# extend pandas functionality with metrics, etc.
qs.extend_pandas()

# fetch the daily returns for a stock
stock = yf.download('META', start='2020-01-01', end='2023-01-01', auto_adjust=False)['Adj Close'].pct_change().dropna()


# show sharpe ratio
qs.stats.sharpe(stock)

# or using extend_pandas() :)
stock.sharpe()


qs.reports.full(stock, "SPY")



qs.plots.snapshot(stock, title='Facebook Performance', show=True)

import quantstats as qs
import yfinance as yf

# Estende a funcionalidade do pandas com métricas, etc.
qs.extend_pandas()

# Obtém os retornos diários para uma ação
stock = qs.utils.download_returns('META')

# Gera e salva o relatório completo como um arquivo HTML
qs.reports.html(stock, benchmark='SPY', output='relatorio_META.html')



import quantstats as qs
import yfinance as yf
import matplotlib
import pandas as pd
import numpy as np
import webbrowser
import os

# Usar backend não-interativo para evitar erros
matplotlib.use('Agg')

# Estender funcionalidade do pandas com métricas do QuantStats
qs.extend_pandas()

# Definir parâmetros
ticker = 'META'
benchmark = 'SPY'
periodo_inicio = '2020-01-01'
periodo_fim = '2023-01-01'
nome_arquivo = f'{ticker}_relatorio.html'

print(f"Baixando dados para {ticker}...")

# Baixar dados de preço
stock_prices = yf.download(ticker, start=periodo_inicio, end=periodo_fim, auto_adjust=True)['Close']

# IMPORTANTE: Converter preços para retornos percentuais diários
# Esta é a correção principal para o seu erro
stock_returns = stock_prices.pct_change().dropna()

# Verificar se temos dados suficientes
if len(stock_returns) < 20:
    print("ERRO: Poucos dados disponíveis. Verifique o ticker e o período.")
    exit()

print(f"Dados baixados com sucesso! {len(stock_returns)} dias de dados.")

# Mostrar algumas estatísticas básicas
print("\nEstatísticas básicas:")
sharpe = qs.stats.sharpe(stock_returns)
print(f"Sharpe Ratio: {sharpe.iloc[0]:.4f}")
cagr = qs.stats.cagr(stock_returns)
print(f"CAGR: {cagr:.2%}")
max_dd = qs.stats.max_drawdown(stock_returns)
print(f"Máximo Drawdown: {max_dd:.2%}")

# Agora vamos gerar o relatório completo
print(f"\nGerando relatório completo para {ticker}...")

try:
    # Usando o método html em vez de full para salvar diretamente como arquivo
    qs.reports.html(stock_returns, benchmark=benchmark, 
                    title=f'Relatório de Performance: {ticker}',
                    output=nome_arquivo)
    
    print(f"Relatório gerado com sucesso e salvo como {nome_arquivo}")
    
    # Abrir o relatório no navegador
    caminho_absoluto = os.path.abspath(nome_arquivo)
    url = f"file://{caminho_absoluto}"
    
    # Perguntar se quer abrir o relatório
    resposta = input("Deseja abrir o relatório no navegador? (s/n): ")
    if resposta.lower() in ['s', 'sim', 'y', 'yes']:
        print("Abrindo no navegador...")
        webbrowser.open(url)
    
except Exception as e:
    print(f"Erro ao gerar relatório: {str(e)}")
    
    # Se falhou com html(), vamos tentar alguns gráficos individuais
    print("\nGerando gráficos individuais como alternativa...")
    
    # Criar pasta para gráficos
    pasta_graficos = 'graficos_quantstats'
    if not os.path.exists(pasta_graficos):
        os.makedirs(pasta_graficos)
    
    # Gerar alguns gráficos básicos e salvá-los
    graficos = [
        ('returns', 'Retornos Cumulativos'),
        ('monthly_heatmap', 'Heatmap Mensal'),
        ('drawdown', 'Drawdowns'),
        ('monthly_returns', 'Retornos Mensais'),
        ('yearly_returns', 'Retornos Anuais')
    ]
    
    for func_nome, titulo in graficos:
        try:
            plt.figure(figsize=(10, 6))
            plot_func = getattr(qs.plots, func_nome)
            plot_func(stock_returns, title=f'{ticker} - {titulo}')
            nome_arquivo_graf = os.path.join(pasta_graficos, f'{ticker}_{func_nome}.png')
            plt.savefig(nome_arquivo_graf)
            plt.close()
            print(f"Gráfico '{titulo}' salvo em {nome_arquivo_graf}")
        except Exception as e:
            print(f"Não foi possível gerar o gráfico '{titulo}': {str(e)}")
            plt.close()
