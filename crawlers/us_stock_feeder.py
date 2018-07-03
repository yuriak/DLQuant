# -*- coding:utf-8 -*-
from pymongo import MongoClient
import dateutil.parser as parser
import numpy as np
from pymongo.errors import BulkWriteError, DuplicateKeyError
import sys
from utils.ConfigHandler import ConfigHandler

# ===================================================================================================
# please use this address to download the dataset and use this code to feed the data into mongodb
# https://www.quandl.com/databases/WIKIP/usage/export
# ===================================================================================================

MONGO_URL = None
MONGO_USERNAME = None
MONGO_PASSWORD = None
config = {
    "MONGO_URL": MONGO_URL,
    "MONGO_USERNAME": MONGO_USERNAME,
    "MONGO_PASSWORD": MONGO_PASSWORD
}
config_handler = ConfigHandler(config_path='../configs/us_stock_crawler_config.json', default_config=config)

if not config_handler.is_config_exist():
    config_handler.write_config(config)
else:
    config = config_handler.read_config()
    MONGO_URL = config['MONGO_URL']
    MONGO_USERNAME = config['MONGO_USERNAME']
    MONGO_PASSWORD = config['MONGO_PASSWORD']

client = MongoClient(MONGO_URL)
db = client.stockdb
db.authenticate(name=MONGO_USERNAME, password=MONGO_PASSWORD)
stock_coll = db.stockcoll

if __name__ == '__main__':
    # please download the dataset, unzip and input the csv path
    path = sys.argv[1]
    with open(path) as f:
        for line in f.readlines()[1:]:
            d = line.strip().split(',')
            _id = '%s_%s' % (d[0], d[1])
            symbol = d[0]
            date = parser.parse(d[1])
            open_price = float(d[2]) if d[2] != '' else np.nan
            high = float(d[3]) if d[3] != '' else np.nan
            low = float(d[4]) if d[4] != '' else np.nan
            close = float(d[5]) if d[5] != '' else np.nan
            volume = float(d[6]) if d[6] != '' else np.nan
            ex_dividend = float(d[7]) if d[7] != '' else np.nan
            split_ratio = float(d[8]) if d[8] != '' else np.nan
            adj_open = float(d[9]) if d[9] != '' else np.nan
            adj_high = float(d[10]) if d[10] != '' else np.nan
            adj_low = float(d[11]) if d[11] != '' else np.nan
            adj_close = float(d[12]) if d[12] != '' else np.nan
            adj_volume = float(d[13]) if d[13] != '' else np.nan
            try:
                stock_coll.insert_one({'_id': _id, 'symbol': symbol, 'date': date, 'open': open_price, 'high': high, 'low': low,
                                       'close': close, 'volume': volume, 'ex_dividend': ex_dividend, 'split_ratio': split_ratio, 'adj_open': adj_open, 'adj_high': adj_high,
                                       'adj_low': adj_low, 'adj_close': adj_close, 'adj_volume': adj_volume})
            except DuplicateKeyError as b:
                pass
