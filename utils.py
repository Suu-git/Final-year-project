from tensorflow import keras
import pandas_datareader as web
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import *
import matplotlib.pyplot as plt
import pickle

class StockPredictionModel():
  def __init__(self, my_model, scaler):
    self.LSTM_model = keras.models.load_model(my_model)
    self.N = self.LSTM_model.input.shape[1]
    with open(scaler, 'rb') as f:
      self.scaler = pickle.load(f)
  def predict(self, dataFrame, numberOfDays):
    """
    dataFrame = a pandas data frame with 5 columns: Open, High,	Low,	Close and Volumns	
    """
    try:
      if dataFrame.keys().tolist() != ['Open', 'High', 'Low', 'Close', 'Volume']:
        return "Please the order of the columns's name"
    except:
      return False
    data = dataFrame.to_numpy()
    data = self.scaler.transform(data)
    y = []
    for _ in range(numberOfDays):
      data = data[len(data)-30:,:]
      result = self.LSTM_model.predict(np.array([data]))
      data = np.concatenate((data, result), axis=0)
      y.append(result.tolist()[0])
    result = self.scaler.inverse_transform(np.array(y))
    result = pd.DataFrame(np.array(result), columns =['Open', 'High', 'Low', 'Close', 'Volume'])
    return result


DogeModel = StockPredictionModel('LSTM_Doge','scaler_Doge.pkl')
BitCoinModel = StockPredictionModel('LSTM_Bitcoin','scaler_Bitcoin.pkl')
ETHModel = StockPredictionModel('LSTM_ETH','scaler_ETH.pkl')