# -*- coding:utf-8 -*-
from urllib import request
import json
import dateutil.parser
import datetime
import re
import pandas as pd
import os
from pymongo import MongoClient
from pymongo.errors import BulkWriteError, DuplicateKeyError
import time
from utils.ConfigHandler import ConfigHandler
import csv

# If you want to use this code, you'd better have a mongodb server
# Because the file storing feature has not been tested, sorry about that :)
# Or, you can implement your own storage method

FILE_PATH = 'reuters_news.csv'
MONGO_URL = None
MONGO_USERNAME = None
MONGO_PASSWORD = None
HEADLINE_URL = 'http://mobile.reuters.com/assets/jsonHeadlines?channel=113&limit=5'
NEWS_URL = 'http://www.reuters.com/article/json/data-id'

config = {
    "FILE_PATH": FILE_PATH,
    "MONGO_URL": MONGO_URL,
    "MONGO_USERNAME": MONGO_USERNAME,
    "MONGO_PASSWORD": MONGO_PASSWORD
}

config_handler = ConfigHandler(config_path='../configs/reuters_crawler_config.json', default_config=config)

if not config_handler.is_config_exist():
    config_handler.write_config(config)
else:
    config = config_handler.read_config()
    FILE_PATH = config['FILE_PATH']
    MONGO_URL = config['MONGO_URL']
    MONGO_USERNAME = config['MONGO_USERNAME']
    MONGO_PASSWORD = config['MONGO_PASSWORD']


def headline_url(last_time):
    if last_time == '':
        return HEADLINE_URL
    else:
        return HEADLINE_URL + '&endTime=' + last_time


def news_url(news_id):
    return NEWS_URL + news_id


def dump_news(news, collection, csv_writer):
    news_str = list(map(lambda x: str(x), news.values()))
    csv_writer.writerow(news_str)
    if collection is not None:
        collection.insert_one(news)


news_coll = None
if (MONGO_URL and MONGO_USERNAME and MONGO_PASSWORD) is not None:
    client = MongoClient(MONGO_URL)
    db = client.stockdb
    db.authenticate(name=MONGO_USERNAME, password=MONGO_PASSWORD)
    news_coll = db.news_latest

res = request.urlopen(headline_url(''))
headlines = json.loads(res.read().decode())['headlines']

lastTime = headlines[-1]['dateMillis']
with open(FILE_PATH, 'a+') as f:
    writer = csv.writer(f)
    while lastTime != None:
        saved = 0
        for headline in headlines:
            try:
                news_res = request.urlopen(news_url(headline['id']))
                news_json = json.loads(news_res.read().decode())['story']
                content = re.sub(r'</?\w+[^>]*>', ' ', news_json['body']).replace('\n', ' ')
                published = dateutil.parser.parse(str(datetime.datetime.fromtimestamp(news_json['published']).date()))
                title = news_json['headline']
                source = NEWS_URL + headline['id']
                news = {'_id': headline['id'], 'title': title, 'date': published, 'content': content, 'url': source}
                dump_news(news, news_coll, writer)
                lastTime = headline['dateMillis']
                saved += 1
            except Exception as e:
                lastTime = headline['dateMillis']
                continue
        print('Crawled at %s = %s saved %d' % (headlines[-1]['dateMillis'], str(datetime.datetime.fromtimestamp(int(headlines[-1]['dateMillis']) / 1000)), saved))
        time.sleep(5)
        res = request.urlopen(headline_url(lastTime))
        headlines = json.loads(res.read().decode())['headlines']
