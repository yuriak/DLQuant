import datetime
import os

import pandas as pd
import tushare as ts
from pymongo import MongoClient
from pymongo.errors import BulkWriteError, DuplicateKeyError
import logging

from utils.ConfigHandler import ConfigHandler

START_DATE = '2001-01-01'
MONGO_URL = None
MONGO_USERNAME = None
MONGO_PASSWORD = None
config = {
    "START_DATE": START_DATE,
    "MONGO_URL": MONGO_URL,
    "MONGO_USERNAME": MONGO_USERNAME,
    "MONGO_PASSWORD": MONGO_PASSWORD
}
config_handler = ConfigHandler(config_path='../configs/china_stock_crawler_config.json', default_config=config)

if not config_handler.is_config_exist():
    config_handler.write_config(config)
else:
    config = config_handler.read_config()
    START_DATE = config['START_DATE']
    MONGO_URL = config['MONGO_URL']
    MONGO_USERNAME = config['MONGO_USERNAME']
    MONGO_PASSWORD = config['MONGO_PASSWORD']

stock_coll = None
index_coll = None
if (MONGO_URL and MONGO_USERNAME and MONGO_PASSWORD) is not None:
    client = MongoClient(MONGO_URL)
    db = client.stockdb
    db.authenticate(name=MONGO_USERNAME, password=MONGO_PASSWORD)
    stock_coll = db.stockcoll
    index_coll = db.indexcoll

all_stock = ts.get_stock_basics()
all_index = ts.get_index()
stock_codes = list(filter(lambda x: x[0] != '3', all_stock.index.values))
index_codes = list(all_index.code)

for c in stock_codes:
    try:
        data = ts.bar(code=c, start_date=START_DATE)
        data['date'] = data.index.values
        data['_id'] = data[['code', 'date']].apply(axis=1, func=lambda x: '%s_%s' % (x.code, str(x.date.date())))
        saved = 0
        for k, v in data.iterrows():
            try:
                if v.open == 0 or v.close == 0 or v.high == 0 or v.low == 0 or v.amount < 10:
                    print('%s stop at %s' % (v.code, str(v.date.date())))
                    continue
                stock_coll.insert_one(v.to_dict())
                saved += 1
            except DuplicateKeyError as dke:
                continue
        print("Feed stock complate: %s saved %d" % (c, saved))
    except Exception as e:
        print("Error")
        continue

for c in index_codes:
    try:
        data = ts.bar(code=c, asset='INDEX', start_date=START_DATE)
        data['date'] = data.index.values
        data['_id'] = data[['code', 'date']].apply(axis=1, func=lambda x: '%s_%s' % (x.code, str(x.date.date())))
        saved = 0
        for k, v in data.iterrows():
            try:
                if v.open == 0 or v.close == 0 or v.high == 0 or v.low == 0 or v.amount < 10:
                    continue
                index_coll.insert_one(v.to_dict())
                saved += 1
            except DuplicateKeyError as dke:
                continue
        print("Feed index complate: %s saved %d" % (c, saved))
    except Exception as e:
        print("Error")
        continue
