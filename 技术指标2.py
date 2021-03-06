# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 11:30:47 2016

@author: Ge
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import talib
from backtest import Strategy, Portfolio, getSymbol, Bars, MarketOpenPortfolio ,Performance
import warnings
warnings.filterwarnings("ignore")



class MAStrategy(Strategy):
    
    def __init__(self, symbol, bars_close, long = 24,short = 1):
        self.symbol = symbol   	
        self.bars_close = bars_close
        self.long = long
        self.short = short
    
    def generate_signals(self):
        Cc = self.bars_close
        signals = pd.DataFrame(columns = self.symbol,index = Cc.index)
        signals[self.symbol[0]] = np.where(pd.rolling_mean(Cc[self.symbol[0]],self.short)>pd.rolling_mean(Cc[self.symbol[0]],self.long),1,0)
        return signals
        
        
class RSIStrategy(Strategy):
    
    def __init__(self, symbol, bars_close, look_back = 14,up_line = 70,low_line = 30):
        self.symbol = symbol   	
        self.bars_close = bars_close
        self.look_back = look_back
        self.up_line = up_line
        self.low_line = low_line
        
    def RSI(self):
        Cc = self.bars_close
        RSI = pd.DataFrame(index = Cc.index)
        RSI[self.symbol[0]] = talib.RSI(Cc[self.symbol[0]].values,self.look_back)
        return RSI
    
    def generate_signals(self):
        RSI = self.RSI()
        signals = pd.DataFrame(columns = self.symbol,index = RSI.index)
        signals[self.symbol[0]] = np.where(RSI[self.symbol[0]]<self.low_line,1,np.where(RSI[self.symbol[0]]>self.up_line,0,np.nan))
        signals = signals.fillna(method = "ffill")
        signals = signals.fillna(0)
        return signals
        
class BBANDSStrategy(Strategy):
    
    def __init__(self, symbol, bars_close, look_back = 20, up_multiplier = 2,low_multiplier = 2):
        self.symbol = symbol   	
        self.bars_close = bars_close
        self.look_back = look_back
        self.up_multiplier = up_multiplier
        self.low_multiplier = low_multiplier
        
    def BBANDS(self):
        Cc = self.bars_close
        BBANDS = pd.DataFrame(index = Cc.index)
        BBANDS["u"],BBANDS["m"],BBANDS["l"] = talib.BBANDS(Cc[self.symbol[0]].values,self.look_back,self.up_multiplier,self.low_multiplier)
        return BBANDS
    
    def generate_signals(self):
        BBANDS = self.BBANDS()
        C = self.bars_close
        signals = pd.DataFrame(columns = self.symbol,index = BBANDS.index)
        signals[self.symbol[0]] = np.where(C[self.symbol[0]]>BBANDS["u"],1,np.where(C[self.symbol[0]]<BBANDS["l"],0,np.nan))
        signals = signals.fillna(method = "ffill")
        signals = signals.fillna(0)
        return signals
        
class RSIMAStrategy(Strategy):
    
    def __init__(self, symbol, bars_close, look_back = 14,up_line = 70,low_line = 30,ma_look_back=250):
        self.symbol = symbol   	
        self.bars_close = bars_close
        self.look_back = look_back
        self.up_line = up_line
        self.low_line = low_line
        self.ma_look_back = ma_look_back
        
    def RSI(self):
        Cc = self.bars_close
        RSI = pd.DataFrame(index = Cc.index)
        RSI[self.symbol[0]] = talib.RSI(Cc[self.symbol[0]].values,self.look_back)
        return RSI
    
    def MA(self):
        Cc = self.bars_close
        MA = pd.DataFrame(index = Cc.index)
        MA[self.symbol[0]] = pd.rolling_mean(Cc,self.ma_look_back)
        return MA
    
    def generate_signals(self):
        RSI = self.RSI()
        MA = self.MA()
        signals = pd.DataFrame(columns = self.symbol,index = RSI.index)
        signals[self.symbol[0]] = np.where(MA[self.symbol[0]]<MA[self.symbol[0]].shift(1),np.where(RSI[self.symbol[0]]<self.low_line,1,np.where(RSI[self.symbol[0]]>self.up_line-20,0,np.nan)),np.where(RSI[self.symbol[0]]<self.low_line,1,np.where(RSI[self.symbol[0]]>self.up_line,0,np.nan)))
        signals = signals.fillna(method = "ffill")
        signals = signals.fillna(0)
        return signals
        
    
if __name__ == "__main__":
    symbol = ["SPY"]    
#    O = Bars(symbol).Open(start='2006-01-01')
#    H = Bars(symbol).high(start='2006-01-01')
#    L = Bars(symbol).low(start='2006-01-01')
#    C = Bars(symbol).close(start='2006-01-01')
    bars = getSymbol(symbol[0]).adj_history(start="2000-01-01")
    O = pd.DataFrame()
    H = pd.DataFrame()
    L = pd.DataFrame()
    C = pd.DataFrame()
    V = pd.DataFrame()    
    O[symbol[0]] = bars["open"]
    H[symbol[0]] = bars["high"]
    L[symbol[0]] = bars["low"]
    C[symbol[0]] = bars["close"]
    V[symbol[0]] = bars["volume"]
#    PPP  = Performance(MarketOpenPortfolio(RSIMAStrategy(symbol,C),O),"SPY")
#    P_RSI = Performance(MarketOpenPortfolio(RSIStrategy(symbol,C),O),"SPY")