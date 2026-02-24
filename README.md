# stock_bot
A machine learning model trained on historical market data to trade using papermoney
## Overview
* Data Ingestion: Fetches historical bars (price, volume) using Alpaca's market data endpoints.
* Signal Generation: Processes data through a simple MLP network to predict price movement directions.
* Order Execution: Places buy/sell orders automatically based on model confidence using the Alpaca Python SDK.
## Dependencies
* Pytorch
* Pandas
* Numpy
* Alpaca API
* Version: Python 3.x
## Prereqs
* An Alpaca Account (Paper trading is recommended for testing).
## Installation
1. **Clone Repo**
```
  git clone https://github.com/funnymonke0/stock_bot
```
2. **Install Dependencies**
```
  cd simple-isef-scraper
  pip install -r requirements.txt
```
## Model
The core of this bot is a Multi-Layer Perceptron. It is trained on historical features
* Log Volume (20 bar normalized volume)
* Momentum/return (log normalized return from the previous bar)
* Log OHLV (log normalized open, high, low, close based on previous close)
* I chose to keep these inputs simple normalized ohlcv values, but will update at a later date

##Results (confusion matrices, actual accuracies vary around 40%, which I have not graphed yet)
[Results tab](./results)
![model 1.6 confusion matrix](./results/1.6_results.png)
