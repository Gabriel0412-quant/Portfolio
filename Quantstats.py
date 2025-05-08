# QuantStats is a Python library for performance and risk analytics of financial portfolios.
import quantstats as qs
import yfinance as yf

# extend pandas functionality with metrics, etc.
qs.extend_pandas()

# fetch the daily returns for a stock
stock = qs.utils.download_returns('VALE3.SA')


# show sharpe ratio
qs.stats.sharpe(stock)

# or using extend_pandas() :)
stock.sharpe()


qs.plots.snapshot(stock, title='Facebook Performance', show=True)

