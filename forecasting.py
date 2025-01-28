import pandas as pd
from statsmodels.tsa.arima.model import ARIMA 
import matplotlib as plt

def forecasting(data):
    data.set_index('Date', inplace=True)
    data = data['AveragePrice']
