# Applying Deep Learning and NLP in Quantitative Trading

## Overview

This repository is the preliminary work of [RLQuant](https://github.com/yuriak/RLQuant) including 5 data crawlers and 2 models.  

## Crawlers
- China stock crawler 
    - Based on [tushare](https://github.com/waditu/tushare)
    - daily stock and index data  
- Sina news crawler
    - Based on [新浪财经24小时新闻(Sina financial news)](http://live.sina.com.cn/zt/f/v/finance/globalnews1)
    - Securities and companies short news
- Open-Europe crawler
    - Based on [Open Europe Daily News](https://openeurope.org.uk/today/daily-shakeup/)
- Reuters news crawler
    - Based on [Reuters Financial News](https://mobile.reuters.com/)  

## Models
- [Leverage Financial News to Predict Stock Price Movements Using Word Embeddings and Deep Neural Networks](http://aclweb.org/anthology/N16-1041)  
    - A relatively standard NLP workflow
    - Bag of keywords, Polarity score, Category tag were used
    - Predict the sign of next day's return rate
- [Deep learning for event-driven stock prediction](http://dl.acm.org/citation.cfm?id=2832415.2832572)
    - OpenIE (Information Extraction) for extracting Events(subject, predicate, object)
    - Neural Tensor Network for learning Event Embedding
    - DNN, CNN can be used to predict movements
- RNN Auto-encoder for encoding news titles
    - A sequence 2 sequence AE
    - GRU was used
- [Elmo](https://arxiv.org/abs/1802.05365)
    - Using Elmo for news title encoding
    - please find tf version in [Paper_Review](https://github.com/QuantumAgent/Paper_Review)
    
You can use these code to build your experiment data-source. All the data will be stored in MongoDB by default in my code, you can extend the storage part by yourself.  
Most of the code was developed before 2018, after that I turned over to the reinforcement learning approach [(RLQuant)](https://github.com/yuriak/RLQuant) and almost gave up the predictive models, therefore I cannot guarantee all the code works well here. If you find any bugs, please tell me. 