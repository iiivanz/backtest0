# -*- coding: utf-8 -*-
"""
https://www.quantstart.com/articles/Research-Backtesting-Environments-in-Python-with-pandas
@author: Ge
actually i just copied the code
"""
# backtest.py
from abc import ABCMeta, abstractmethod
import ystockquote
import pandas as pd
import numpy as np
import time
today = time.strftime('%Y-%m-%d',time.localtime(time.time()))
class Strategy(object):
    """Strategy is an abstract base class providing an interface for
    all subsequent (inherited) trading strategies.

    The goal of a (derived) Strategy object is to output a list of signals,
    which has the form of a time series indexed pandas DataFrame.

    In this instance only a single symbol/instrument is supported."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_signals(self):
        """An implementation is required to return the DataFrame of symbols 
        containing the signals to go long, short or hold (1, -1 or 0)."""
        raise NotImplementedError("Should implement generate_signals()!")

class Portfolio(object):
    """An abstract base class representing a portfolio of 
    positions (including both instruments and cash), determined
    on the basis of a set of signals provided by a Strategy."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_positions(self):
        """Provides the logic to determine how the portfolio 
        positions are allocated on the basis of forecasting
        signals and available cash."""
        raise NotImplementedError("Should implement generate_positions()!")

    @abstractmethod
    def backtest_portfolio(self):
        """Provides the logic to generate the trading orders
        and subsequent equity curve (i.e. growth of total equity),
        as a sum of holdings and cash, and the bar-period returns
        associated with this curve based on the 'positions' DataFrame.

        Produces a portfolio object that can be examined by 
        other classes/functions."""
        raise NotImplementedError("Should implement backtest_portfolio()!")

class getSymbol(object):
    today = time.strftime('%Y-%m-%d',time.localtime(time.time()))
    def __init__(self, symbol):
        self.symbol = symbol
   
    def adj_history(self,start = '2000-01-01',end = today):
        A = pd.DataFrame(pd.DataFrame(ystockquote.get_historical_prices(self.symbol, start, end)).T)
        A[['Adj Close','Close','High','Low','Open','Volume']] = A[['Adj Close','Close','High','Low','Open','Volume']].convert_objects(convert_numeric=True)
        p = A['Adj Close'] / A['Close']
        A['Adj High'] = p * A['High']
        A['Adj Low'] = p * A['Low']
        A['Adj Open'] = p * A['Open']
        B = pd.DataFrame()
        B['open'] = A['Adj Open']
        B['high'] = A['Adj High']
        B['low'] = A['Adj Low']
        B['close'] = A['Adj Close']
        B['volume'] = A['Volume']
        return B
        
    def history(self,start = '2000-01-01',end = today):
        A = pd.DataFrame(pd.DataFrame(ystockquote.get_historical_prices(self.symbol, start, end)).T)
        A[['Adj Close','Close','High','Low','Open','Volume']] = A[['Adj Close','Close','High','Low','Open','Volume']].convert_objects(convert_numeric=True)
        B = pd.DataFrame()
        B['open'] = A['Open']
        B['high'] = A['High']
        B['low'] = A['Low']
        B['close'] = A['Close']
        B['volume'] = A['Volume']
        B["adj_close"] = A["Adj Close"]
        return B   

class Bars(object):
    today = time.strftime('%Y-%m-%d',time.localtime(time.time()))
    def __init__(self,symbol):
        self.symbol = symbol
                
    def Open(self,start = '2000-01-01',end = today):
        bar = pd.DataFrame()
        for x in self.symbol:
            bar[x] = getSymbol(x).adj_history(start = start,end = today)['open']
        bars_open = bar.dropna()  
        return bars_open
        
    def close(self,start = '2000-01-01',end = today):
        bar = pd.DataFrame()
        for x in self.symbol:
            bar[x] = getSymbol(x).adj_history(start = start,end = today)['close']
        bars_close = bar.dropna()  
        return bars_close
        
    def high(self,start = '2000-01-01',end = today):
        bar = pd.DataFrame()
        for x in self.symbol:
            bar[x] = getSymbol(x).adj_history(start = start,end = today)['high']
        bars_high = bar.dropna()  
        return bars_high    
        
    def low(self,start = '2000-01-01',end = today):
        bar = pd.DataFrame()
        for x in self.symbol:
            bar[x] = getSymbol(x).adj_history(start = start,end = today)['low']
        bars_low = bar.dropna()  
        return bars_low  
        
    def volume(self,start = '2000-01-01',end = today):
        bar = pd.DataFrame()
        for x in self.symbol:
            bar[x] = getSymbol(x).adj_history(start = start,end = today)['volume']
        bars_volume = bar.dropna()  
        return bars_volume
        
class MarketOpenPortfolio(Portfolio):
    
    def __init__(self, Strategy, bars_bid):
        self.symbol = Strategy.symbol        
        self.bars_bid = bars_bid
        self.signals = Strategy.generate_signals()
        self.positions = self.generate_positions()

    def generate_positions(self):
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions = self.signals
        return positions    
              
    def backtest_portfolio(self):        
        bid = self.bars_bid
        bars_returns = pd.DataFrame(index = bid.index)
        bars_returns = bid / bid.shift(1) - 1
        bars_returns = (self.positions.shift(1) * bars_returns.shift(-1))
        portfolio = pd.DataFrame()
        portfolio = bars_returns
        portfolio['holdings'] = ((bars_returns).sum(axis = 1) + 1).cumprod()
        portfolio['holdings'][0] = 1
        return portfolio
        
        
class MarketClosePortfolio(Portfolio):
    
    def __init__(self, Strategy, bars_bid):
        self.symbol = Strategy.symbol        
        self.bars_bid = bars_bid
        self.signals = Strategy.generate_signals()
        self.positions = self.generate_positions()

    def generate_positions(self):
        positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        positions = self.signals
        return positions    
              
    def backtest_portfolio(self):        
        bid = self.bars_bid
        bars_returns = pd.DataFrame(index = bid.index)
        bars_returns = bid / bid.shift(1) - 1
        bars_returns = (self.positions.shift(1) * bars_returns.shift(-1))
        portfolio = pd.DataFrame()
        portfolio = bars_returns
        portfolio['holdings'] = ((bars_returns).sum(axis = 1) + 1).cumprod()
        portfolio['holdings'][0] = 1
        return portfolio
        
        
class Performance(object):
    
    def __init__(self, Portfolio,benchmark = "SPY", risk_free_rate = 0):
        self.portfolio = Portfolio.backtest_portfolio()
        self.positions = Portfolio.generate_positions()
        self.signals = Portfolio.signals
        self.bars = Portfolio.bars_bid
        if benchmark == None:
            self.benchmark = "NaN"
        else:
            self.benchmark = getSymbol(benchmark).adj_history(start = self.portfolio['holdings'].index[0],end = self.portfolio['holdings'].index[-1] )['close']
        self.risk_free_rate = risk_free_rate
        
    def total_return(self):
        total_return = self.portfolio['holdings'][-1] - 1
        return total_return
        
    def annual_return(self):
        period = len(self.portfolio['holdings'])
        total_return = self.total_return()
        annual_return = (total_return + 1) ** (250 / period) - 1
        return annual_return
        
    def risk(self):
        std = self.portfolio['holdings'].pct_change().std()
        risk = std * np.sqrt(250)
        return risk
 
    def benchmark_return(self):
        return self.benchmark.pct_change()
        
    def drawdown(self):
        drawdown = self.portfolio['holdings'] / pd.expanding_max(self.portfolio['holdings']) - 1
        return drawdown
         
    def max_drawdown(self):
        drawdown = self.drawdown()
        max_drawdown = drawdown.min()
        end_date = drawdown.idxmin()
        start_date = drawdown[:end_date][drawdown == 0].index[-1]
        if pd.to_datetime(end_date) == pd.to_datetime(self.portfolio['holdings'].index[-1]):
            end_date = "NaN"
        return max_drawdown,start_date,end_date
        
    def beta(self):
        beta = self.benchmark_return().cov(self.portfolio['holdings'].pct_change()) / self.benchmark_return().var()
        return beta
        
    def alpha(self):
        benchmark_price = self.benchmark
        benchmark_annual_return = (benchmark_price[-1] / benchmark_price[0]) ** (250 / len(benchmark_price.index)) - 1
        alpha = (self.annual_return() - self.risk_free_rate) - self.beta() * (benchmark_annual_return - self.risk_free_rate)
        return alpha
        
    def sharpe_ratio(self):
        sharpe_ratio = (self.annual_return() - self.risk_free_rate) / self.risk()
        return sharpe_ratio
    
    def show(self):
        d,s,e = self.max_drawdown()
        pd.to_datetime(e) - pd.to_datetime(s)
        Show = pd.Series()
        Show["annual return"] = self.annual_return()
        Show["risk"] = self.risk()
        Show["rr ratio"] = self.annual_return() / self.risk()
        Show["max drawdown"] = d
        Show["max drawdown duration"] = (str((pd.to_datetime(e) - pd.to_datetime(s)).days) + str(' days'))
        Show["beta"] = self.beta()
        Show["alpha"] = self.alpha()
        Show["sharpe ratio"] = self.sharpe_ratio()
        com = pd.DataFrame(columns = [['Portfolio','Benchmark']])
        com['Portfolio'] = self.portfolio['holdings']
        com['Benchmark'] = self.benchmark / self.benchmark[0]
        com.plot(figsize=(14,8))
        print("策略回测结果：")        
        return Show
        
    def sim(self):
        signals = self.signals
        C = self.bars
        positions = signals.shift(1).abs()
        positions.iloc[0] = 0
        Cop = pd.DataFrame(columns = positions.columns,index = positions.index)
        Cop[Cop.columns] = np.where((positions - positions.shift(1))==1,np.where(signals.shift(1)>0,1,2),np.where((positions - positions.shift(1))==-1,np.where(signals.shift(3)>0,-1,-2),0))
        sim = pd.DataFrame()
        sim_tem = pd.DataFrame(columns=[["symbol","operation","start_date","end_date","take","close"]])
        for x in signals.columns:
            take = list(Cop[Cop[x]>0].dropna(how = "all").index)
            close = list(Cop[Cop[x]<0].dropna(how = "all").index)
            if len(take) > len(close):
                close.append(np.nan)
            sim_tem["start_date"] = take
            sim_tem["end_date"] = close
            sim_tem["symbol"] = x
            sim = pd.concat([sim,sim_tem])
        sim.index = range(len(sim.index))
        for x in range(len(sim.index)):
            if Cop[sim["symbol"][x]][sim["start_date"][x]] == 1: 
                sim["operation"][x] = 1
            else:
                sim["operation"][x] = -1
            sim["take"][x] = C[sim["symbol"][x]][sim["start_date"][x]]
            if sim["end_date"].isnull()[x]:
                sim["close"][x] = C[sim["symbol"][x]].iloc[-1]
            else:
                sim["close"][x] = C[sim["symbol"][x]][sim["end_date"][x]] 
        sim["P&L"] = (sim["close"] / sim["take"]) ** sim["operation"] - 1
        sim["holding_days"] = pd.to_datetime(sim["end_date"]) - pd.to_datetime(sim["start_date"])
#        result = pd.DataFrame()
        sim = sim.sort(["start_date"])
        sim.index = range(len(sim.index))
        return sim
        
    def sim_summary(self,com = 0):
        sim = self.sim()
        S = pd.Series()
        S["total_return"] = (sim["P&L"]+1).prod() - 1
        S["trade_frequency"] = len(sim.index)
        S["win_rate"] = len(sim[sim["P&L"]>com].index) / len(sim.index)
#        S["average_hold"] = 
        return S