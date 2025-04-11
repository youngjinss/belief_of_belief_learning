import datetime
from binance_historical_data import BinanceDataDumper

data_dumper = BinanceDataDumper(
    path_dir_where_to_dump="./data/binance/",
    asset_class="um",  # spot, um, cm
    data_type="klines",  # aggTrades, klines, trades
    data_frequency="30m",
)

data_dumper.dump_data(
    tickers=["BTCUSDT"],
    date_start=datetime.date(2025, 2, 1),
    date_end=datetime.date(2025, 4, 1),
    is_to_update_existing=True,
)
