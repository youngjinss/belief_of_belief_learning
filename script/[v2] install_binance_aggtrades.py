import datetime
from binance_historical_data import BinanceDataDumper

data_dumper = BinanceDataDumper(
    path_dir_where_to_dump="./data/binance/",
    asset_class="um",  # spot, um, cm
    data_type="aggTrades",  # aggTrades, klines, trades
    data_frequency="1h",
)

data_dumper.dump_data(
    tickers=["BTCUSDT"],
    date_start=datetime.date(2023, 10, 1),
    date_end=None,
    is_to_update_existing=True,
)
