# -*- coding:utf-8 -*-
from urllib import request
import requests
import json
from bs4 import BeautifulSoup
from dateutil.parser import parse
from pymongo import MongoClient
from pymongo.errors import BulkWriteError, DuplicateKeyError
import sys

from utils.ConfigHandler import ConfigHandler

MONGO_URL = None
MONGO_USERNAME = None
MONGO_PASSWORD = None
# db_name: stockdb
# collection: europe_news
# old news file: euro_old_news.txt and euro_old_news_json.json

# If you want to use this code, you'd better have a mongodb server
# Because the file storing feature has not been tested, sorry about that :)
# Or, you can implement your own storage method


config = {
    "MONGO_URL": MONGO_URL,
    "MONGO_USERNAME": MONGO_USERNAME,
    "MONGO_PASSWORD": MONGO_PASSWORD
}
config_handler = ConfigHandler(config_path='../configs/europe_crawler_config.json', default_config=config)
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
    news_coll = db.europe_news


def crawl_news(url, date):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'}
    response = requests.get(url, headers=headers)
    bs = BeautifulSoup(response.text, 'lxml')
    head_title = bs.select('h1')[0].text
    head_content = bs.select('h1')[0].find_next_sibling().text
    news_date_str = date.strftime('%Y-%m-%d')
    news_id = 0
    try:
        news_coll.insert_one({'_id': '{0}_{1}'.format(news_date_str, news_id), 'title': head_title, 'date': date, 'content': head_content})
    except Exception as e:
        pass
    for section_news in bs.select('.section'):
        news_id += 1
        section_title = section_news.find('h2').text
        if section_news.find('p') is not None:
            section_content = section_news.find('p').text
        else:
            section_content = ''
        try:
            news_coll.insert_one({'_id': '{0}_{1}'.format(news_date_str, news_id), 'title': section_title, 'date': date, 'content': section_content})
        except Exception as e:
            pass
    print(date)


def crawl_news_url(year, month, page=1):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'}
    response = requests.get('https://openeurope.org.uk/today/media-digest/{0}/{1}/?pp={2}&ajax'.format(year, month, page), headers=headers)
    response_json = response.json()
    bs = BeautifulSoup(response_json['html'], 'lxml')
    urls = []
    for url_info in bs.select('.md-date'):
        d = url_info.select('.digest-day')[0].text
        print(year, month, d)
        date = parse('{0}-{1}-{2}'.format(year, month, d))
        url = url_info.get('href')
        urls.append((url, date))
    if response_json['more'] is not None:
        urls.extend(crawl_news_url(year, month, page + 1))
    return urls


def crawl_news_from_new_website():
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/55.0.2883.87 Safari/537.36'}
    response = requests.get('https://openeurope.org.uk/today/daily-shakeup/', headers=headers)
    bs = BeautifulSoup(response.text, 'lxml')
    all_dates = []
    for cld in bs.select('.calendar'):
        for addr in cld.find_all('a'):
            year, month = tuple(addr.get('href').split('/')[3:5])
            all_dates.append((year.strip(), month.strip()))
    for y, m in all_dates:
        for url, date in crawl_news_url(y, m):
            crawl_news(url, date)


def crawl_news_from_old_website():
    all_news = []
    with open('euro_old_news.txt', 'w+') as f:
        for i in range(0, 2160, 10):
            url = 'http://archive.openeurope.org.uk/tableNewsFetch/Page/en/LIVE?pgeType=InTheNews&pgeCat=&chrono=&pageCode=InTheNews&sEcho=1&iColumns=3&sColumns=nws_Id%2Cnws_Date%2Cmws_Data&iDisplayStart={0}&iDisplayLength=10&iSortingCols=1&iSortCol_0=0&sSortDir_0=asc&bSortable_0=true&bSortable_1=true&bSortable_2=false&_=1517476692585'.format(
                i)
            response_json = requests.get(url).json()
            for news in response_json['aaData']:
                
                news_date = parse(news[1])
                news_date_str = news_date.strftime('%Y-%m-%d')
                id = '%s_%s' % (news_date_str, str(news[0]))
                bs = BeautifulSoup(news[2], 'lxml')
                title = bs.select('.newsheader')[0].text
                content = bs.select('.newsbody')[0].text
                f.write('{0}\t{1}\t{2}\t{3}\n'.format(id, news_date_str, title, content))
                try:
                    news_coll.insert_one({'_id': id, 'title': title, 'date': news_date, 'content': content})
                except Exception as e:
                    pass
                all_news.append({'id': id, 'date': news_date_str, 'title': title, 'content': content})
            print(i)
    with open('euro_old_news_json.json', 'w+') as f:
        f.write(json.dumps(all_news))


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Please input crawl mode <0,1>")
        sys.exit(0)
    mode = sys.argv[1]
    if mode == '0':
        crawl_news_from_old_website()
    elif mode == '1':
        crawl_news_from_new_website()
