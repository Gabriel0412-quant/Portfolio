import yfinance as yf

from binance.client import Client

Client = Client()

btcbrl = Client.get_klines(symbol= "BTCBRL", interval=Client.KLINE_INTERVAL_1DAY)

btcbrl = [list(map(float, line)) for line in btcbrl]


import jason 

import pandas as pd



for line in btcbrl:
    del line[5:]


btcbrl = pd.DataFrame(btcbrl, columns=["Date", "Open", "High", "Low", "Close"])
btcbrl.set_index("Date", inplace=True)

btcbrl.index = pd.to_datetime(btcbrl.index, unit="ms")

btcbrl['Close'] = pd.to_numeric(btcbrl['Close'])

btcbrl

ethbrl = Client.get_historical_klines("ETHBRL", Client.KLINE_INTERVAL_1DAY, "1 year ago UTC")