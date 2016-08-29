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

class MAStrategy(Strategy):
    
    def __init__(self, symbol, bars_close, long = 24,short = 12):
        self.symbol = symbol   	
        self.bars_close = bars_close
        self.long = long
        self.short = short
    
    def generate_signals(self):
        Cc = self.bars_close
        C = pd.DataFrame()
        C[self.symbol[0]] = Cc
        signals = pd.DataFrame(columns = self.symbol,index = C.index)
        signals[self.symbol[0]] = np.where(pd.rolling_mean(C[self.symbol[0]],self.short)>pd.rolling_mean(C[self.symbol[0]],self.long),1,0)
        return signals
        
class MACD1Strategy(Strategy):
    
    def __init__(self, symbol, bars_close, fastp = 12,slowp = 26,signalp = 9):
        self.symbol = symbol   	
        self.bars_close = bars_close
        self.fastp = fastp
        self.slowp = slowp
        self.signalp = signalp
        
    def MACD(self):
        Cc = self.bars_close
        C = pd.DataFrame()
        MD = pd.DataFrame(index = Cc.index)
        C[self.symbol[0]] = Cc
        macd_line,signal_line,hist = talib.MACD(C[self.symbol[0]].values,self.fastp,self.slowp,self.signalp)
        MD["macd_line"] = macd_line
        MD["signal_line"] = signal_line
        MD['hist'] = hist
        return MD      
                
    def generate_signals(self):
        MD = self.MACD()
        signals = pd.DataFrame(columns = self.symbol,index = MD.index)
        signals[self.symbol[0]] = np.where(MD['hist']>0,1,0)
        return signals  
        
class MACD2Strategy(Strategy):
    
    def __init__(self, symbol, bars_close, fastp = 12,slowp = 26,signalp = 9):
        self.symbol = symbol   	
        self.bars_close = bars_close
        self.fastp = fastp
        self.slowp = slowp
        self.signalp = signalp
        
    def MACD(self):
        Cc = self.bars_close
        C = pd.DataFrame()
        MD = pd.DataFrame(index = Cc.index)
        C[self.symbol[0]] = Cc
        macd_line,signal_line,hist = talib.MACD(C[self.symbol[0]].values,self.fastp,self.slowp,self.signalp)
        MD["macd_line"] = macd_line
        MD["signal_line"] = signal_line
        MD['hist'] = hist
        return MD   
    
    def generate_signals(self):
        MD = self.MACD()
        signals = pd.DataFrame(columns = self.symbol,index = MD.index)
        signals[self.symbol[0]] = np.where(MD['macd_line']>0,1,0)
        return signals
        
class STOCHStrategy(Strategy):
    
    def __init__(self, symbol, bars_high, bars_low, bars_close, fastkp = 5,slowkp = 3,slowdp = 3,take_line = 10,close_line=90):
        self.symbol = symbol 
        self.bars_high = bars_high
        self.bars_low = bars_low
        self.bars_close = bars_close
        self.fastkp = fastkp
        self.slowkp = slowkp
        self.slowdp = slowdp
        self.take_line = take_line
        self.close_line = close_line
    def KD(self):
        H = self.bars_high
        L = self.bars_low
        C = self.bars_close
        fk = self.fastkp
        sk = self.slowkp
        sd = self.slowdp
        slowk,slowd = talib.STOCH(H.values,L.values,C.values,fastk_period=fk,slowk_period=sk,slowd_period=sd)
        return slowk,slowd
        
    def generate_signals(self):
        slowk,slowd = self.KD()
        signals = pd.DataFrame(columns = self.symbol,index = C.index)
        signals[self.symbol[0]] = np.where(slowk<self.take_line,1,np.where(slowd<self.take_line,1,np.where(slowd>self.close_line,0,np.where(slowk>self.close_line,0,np.nan))))
        signals = signals.fillna(method="ffill")
        signals = signals.fillna(0)
        return signals    
    
    
    
if __name__ == "__main__":
    symbol = ["QQQ"]
    
#    O = Bars(symbol).Open(start='2006-01-01')
#    H = Bars(symbol).high(start='2006-01-01')
#    L = Bars(symbol).low(start='2006-01-01')
#    C = Bars(symbol).close(start='2006-01-01')
    
    bars = getSymbol(symbol[0]).adj_history()
    O = Bars(symbol).Open(start='2000-01-01')
    H = bars["high"]
    L = bars["low"]
    C = bars["close"]  
    P_MA = Performance(MarketClosePortfolio(MAStrategy(symbol,C),O),"QQQ")
    P_MACD12269_HIST = Performance(MarketClosePortfolio(MACD1Strategy(symbol,C,12,26,9),O),"QQQ")
    P_MACD5355_HIST = Performance(MarketClosePortfolio(MACD1Strategy(symbol,C,5,35,5),O),"QQQ")       
    P_KD = Performance(MarketClosePortfolio(STOCHStrategy(symbol,H,L,C,fastkp = 9,slowkp = 3,slowdp = 3),O),"QQQ")
