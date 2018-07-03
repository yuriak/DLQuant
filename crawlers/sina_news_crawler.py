# -*- coding:utf-8 -*-
import datetime

from pymongo import MongoClient
from pymongo.errors import BulkWriteError, DuplicateKeyError
from urllib import request
import json
from bs4 import BeautifulSoup
import threading
import time
import logging
from utils.ConfigHandler import ConfigHandler

# If you want to use this code, you'd better have a mongodb server
# Or, you can implement your own storage method


MONGO_URL = None
MONGO_USERNAME = None
MONGO_PASSWORD = None
config = {
    "MONGO_URL": MONGO_URL,
    "MONGO_USERNAME": MONGO_USERNAME,
    "MONGO_PASSWORD": MONGO_PASSWORD
}
config_handler = ConfigHandler(config_path='../configs/sina_crawler_config.json', default_config=config)
if not config_handler.is_config_exist():
    config_handler.write_config(config)
else:
    config = config_handler.read_config()
    MONGO_URL = config['MONGO_URL']
    MONGO_USERNAME = config['MONGO_USERNAME']
    MONGO_PASSWORD = config['MONGO_PASSWORD']

news_coll = None
if (MONGO_URL and MONGO_USERNAME and MONGO_PASSWORD) is not None:
    client = MongoClient(MONGO_URL)
    db = client.stockdb
    db.authenticate(name=MONGO_USERNAME, password=MONGO_PASSWORD)
    news_coll = db.short_news


class CompanyCrawler(threading.Thread):
    def __init__(self, start_id, news_coll):
        threading.Thread.__init__(self)
        self.start_id = start_id
        self.news_coll = news_coll
    
    def run(self):
        try:
            same = 0
            while True:
                if same >= 44:
                    logging.info("Company crawler reached last record, stopping")
                    break
                same = 0
                company_res = request.urlopen('http://live.sina.com.cn/zt/api/f/get/finance/globalnews1/index.htm?format=json&id=%s&tag=3&pagesize=45&dire=b' % (str(self.start_id)))
                json_bytes = company_res.read()
                json_str = json.loads(json_bytes.decode())
                news_json = json_str['result']['data']
                last_date = None
                saved = 0
                for news in news_json:
                    news_date = datetime.datetime.fromtimestamp(int(news['created_at']))
                    news_content = news['content']
                    news_id = news['id']
                    news_tag = news['tag']
                    try:
                        news_coll.insert_one({'_id': news_id, 'content': news_content, 'tag': news_tag, 'date': news_date})
                        saved += 1
                    except DuplicateKeyError as dke:
                        same += 1
                        continue
                    finally:
                        self.start_id = int(news_id)
                        last_date = news_date
                print("Company crawler last feed at %s saved %d" % (str(last_date), saved))
                time.sleep(5)
        except Exception as e:
            logging.error("Company crawler dead")


class SecuritiesCrawler(threading.Thread):
    def __init__(self, start_id, news_coll):
        threading.Thread.__init__(self)
        self.start_id = start_id
        self.news_coll = news_coll
    
    def run(self):
        try:
            same = 0
            while True:
                if same > 44:
                    logging.info("Securities crawler reached last record, stopping")
                    break
                same = 0
                securities_res = request.urlopen('http://live.sina.com.cn/zt/api/f/get/finance/globalnews1/index.htm?format=json&id=%s&tag=10&pagesize=45&dire=b' % (str(self.start_id)))
                json_bytes = securities_res.read()
                json_str = json.loads(json_bytes.decode())
                news_json = json_str['result']['data']
                last_date = None
                saved = 0
                for news in news_json:
                    news_date = datetime.datetime.fromtimestamp(int(news['created_at']))
                    news_content = news['content']
                    news_id = news['id']
                    news_tag = news['tag']
                    try:
                        news_coll.insert_one({'_id': news_id, 'content': news_content, 'tag': news_tag, 'date': news_date})
                        saved += 1
                    except DuplicateKeyError as dke:
                        same += 1
                        continue
                    finally:
                        self.start_id = int(news_id)
                        last_date = news_date
                print("Securities feeder last feed at %s saved %d" % (str(last_date), saved))
                time.sleep(5)
        except Exception as e:
            logging.error("Securities crawler dead")


if __name__ == '__main__':
    try:
        company_res = request.urlopen('http://live.sina.com.cn/zt/f/v/finance/globalnews1?tag=3')
        securities_res = request.urlopen('http://live.sina.com.cn/zt/f/v/finance/globalnews1?tag=10')
        company_bs = BeautifulSoup(company_res.read(), 'lxml')
        securities_bs = BeautifulSoup(securities_res.read(), 'lxml')
        company_latest_id = company_bs.select('.bd_i')[0].attrs['data-id']
        company_latest_id = int(company_latest_id) + 1
        securities_latest_id = securities_bs.select('.bd_i')[0].attrs['data-id']
        securities_latest_id = int(securities_latest_id) + 1
        cFeeder = CompanyCrawler(company_latest_id, news_coll)
        sFeeder = SecuritiesCrawler(securities_latest_id, news_coll)
        cFeeder.start()
        sFeeder.start()
    except Exception as e:
        print(e)
