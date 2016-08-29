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
from backtest import Strategy, Portfolio, getSymbol, Bars, MarketClosePortfolio ,Performance
import warnings
warnings.filterwarnings("ignore")

class VWAPStrategy(Strategy):
    
    def __init__(self, symbol, bars_close, volume, look_back = 3, up_line = 1.001, low_line = 0.9):
        self.symbol = symbol   	
        self.bars_close = bars_close
        self.volume = volume
        self.look_back = look_back
        self.up_line = up_line
        self.low_line = low_line
        
    def VWAP(self):
        Cc = self.bars_close
        Vv = self.volume
        VWAP = pd.DataFrame(index = Cc.index)
        VWAP[self.symbol[0]] = pd.rolling_sum(Cc[self.symbol[0]]*Vv[self.symbol[0]],self.look_back) / pd.rolling_sum(Vv[self.symbol[0]],self.look_back) / Cc[self.symbol[0]]
        return VWAP     
                
    def generate_signals(self):
        VWAP = self.VWAP()
        signals = pd.DataFrame(columns = self.symbol,index = VWAP.index)
        signals[self.symbol[0]] = np.where(VWAP[self.symbol[0]]>self.up_line ,1,0)
        return signals  
    
if __name__ == "__main__":
    symbol = ["SPY"]
    
#    O = Bars(symbol).Open(start='2006-01-01')
#    H = Bars(symbol).high(start='2006-01-01')
#    L = Bars(symbol).low(start='2006-01-01')
#    C = Bars(symbol).close(start='2006-01-01')
    
    bars = getSymbol(symbol[0]).adj_history()
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
    
    
    P_VWAP = Performance(MarketClosePortfolio(VWAPStrategy(symbol,C,V),C),"SPY")