# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 16:26:48 2016

@author: Ge
"""
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sco
import talib
import pandas as pd
import numpy as np
import ystockquote
from backtest import Strategy, Portfolio
import time
from statsmodels.tsa.stattools import coint,adfuller
import warnings
warnings.filterwarnings("ignore")
today = time.strftime('%Y-%m-%d',time.localtime(time.time()))
class etf(object):
    today = time.strftime('%Y-%m-%d',time.localtime(time.time()))
    def __init__(self, symbol):
        self.symbol = symbol
   
    def get_adj_history(self,start = '2000-01-01',end = today):
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
        
    def get_history(self,start = '2000-01-01',end = today):
        A = pd.DataFrame(pd.DataFrame(ystockquote.get_historical_prices(self.symbol, start, end)).T)
        A[['Adj Close','Close','High','Low','Open','Volume']] = A[['Adj Close','Close','High','Low','Open','Volume']].convert_objects(convert_numeric=True)
        B = pd.DataFrame()
        B['open'] = A['Open']
        B['high'] = A['High']
        B['low'] = A['Low']
        B['close'] = A['Close']
        B['volume'] = A['Volume']
        return B   

class Bars(object):
    
    def __init__(self,symbol):
        self.symbol = symbol
                
    def o_pen(self,start = '2000-01-01',end = today):
        bar = pd.DataFrame()
        for x in self.symbol:
            bar[x] = etf(x).get_adj_history(start = start,end = today)['open']
        bars_open = bar.dropna()  
        return bars_open
        
    def close(self,start = '2000-01-01',end = today):
        bar = pd.DataFrame()
        for x in self.symbol:
            bar[x] = etf(x).get_adj_history(start = start,end = today)['close']
        bars_close = bar.dropna()  
        return bars_close
        
    def high(self,start = '2000-01-01',end = today):
        bar = pd.DataFrame()
        for x in self.symbol:
            bar[x] = etf(x).get_adj_history(start = start,end = today)['high']
        bars_high = bar.dropna()  
        return bars_high    
        
    def low(self,start = '2000-01-01',end = today):
        bar = pd.DataFrame()
        for x in self.symbol:
            bar[x] = etf(x).get_adj_history(start = start,end = today)['low']
        bars_low = bar.dropna()  
        return bars_low  
        
    def volume(self,start = '2000-01-01',end = today):
        bar = pd.DataFrame()
        for x in self.symbol:
            bar[x] = etf(x).get_adj_history(start = start,end = today)['volume']
        bars_volume = bar.dropna()  
        return bars_volume
        
        
def calculate_spread_zscore(pairs, symbols, lookback=100):
    """Creates a hedge ratio between the two symbols by calculating
    a rolling linear regression with a defined lookback period. This
    is then used to create a z-score of the 'spread' between the two
    symbols based on a linear combination of the two."""
    
    # Use the pandas Ordinary Least Squares method to fit a rolling
    # linear regression between the two closing price time series
    print("Fitting the rolling Linear Regression...")
    model = pd.ols(y=pairs[symbols[0]], 
                   x=pairs[symbols[1]],
                   window=lookback)

    # Construct the hedge ratio and eliminate the first 
    # lookback-length empty/NaN period
    pairs['hedge_ratio'] = model.beta['x']
    pairs = pairs.dropna()

    # Create the spread and then a z-score of the spread
    print("Creating the spread/zscore columns...")
    pairs['spread'] = pairs[symbols[0]] - pairs['hedge_ratio']*pairs[symbols[1]]
    pairs['zscore'] = (pairs['spread'] - np.mean(pairs['spread']))/np.std(pairs['spread'])
    return pairs
def create_long_short_market_signals(pairs, symbols, 
                                     z_entry_threshold=2.0, 
                                     z_exit_threshold=1.0):
    """Create the entry/exit signals based on the exceeding of 
    z_enter_threshold for entering a position and falling below
    z_exit_threshold for exiting a position."""

    # Calculate when to be long, short and when to exit
    pairs['longs'] = (pairs['zscore'] <= -z_entry_threshold)*1.0
    pairs['shorts'] = (pairs['zscore'] >= z_entry_threshold)*1.0
    pairs['exits'] = (np.abs(pairs['zscore']) <= z_exit_threshold)*1.0

    # These signals are needed because we need to propagate a
    # position forward, i.e. we need to stay long if the zscore
    # threshold is less than z_entry_threshold by still greater
    # than z_exit_threshold, and vice versa for shorts.
    pairs['long_market'] = 0.0
    pairs['short_market'] = 0.0

    # These variables track whether to be long or short while
    # iterating through the bars
    long_market = 0
    short_market = 0

    # Calculates when to actually be "in" the market, i.e. to have a
    # long or short position, as well as when not to be.
    # Since this is using iterrows to loop over a dataframe, it will
    # be significantly less efficient than a vectorised operation,
    # i.e. slow!
    print("Calculating when to be in the market (long and short)...")
    for i, b in enumerate(pairs.iterrows()):
        # Calculate longs
        if b[1]['longs'] == 1.0:
            long_market = 1            
        # Calculate shorts
        if b[1]['shorts'] == 1.0:
            short_market = 1
        # Calculate exists
        if b[1]['exits'] == 1.0:
            long_market = 0
            short_market = 0
        # This directly assigns a 1 or 0 to the long_market/short_market
        # columns, such that the strategy knows when to actually stay in!
        pairs.ix[i]['long_market'] = long_market
        pairs.ix[i]['short_market'] = short_market
    return pairs
def create_portfolio_returns(pairs, symbols):
    """Creates a portfolio pandas DataFrame which keeps track of
    the account equity and ultimately generates an equity curve.
    This can be used to generate drawdown and risk/reward ratios."""
    
    # Convenience variables for symbols
    sym1 = symbols[0]
    sym2 = symbols[1]

    # Construct the portfolio object with positions information
    # Note that minuses to keep track of shorts!
    print("Constructing a portfolio...")
    portfolio = pd.DataFrame(index=pairs.index)
    portfolio['positions'] = pairs['long_market'] - pairs['short_market']
    portfolio[sym1] = -1.0 * pairs[symbols[0]] * portfolio['positions']
    portfolio[sym2] = pairs[symbols[1]] * portfolio['positions']
    portfolio['total'] = portfolio[sym1] + portfolio[sym2]

    # Construct a percentage returns stream and eliminate all 
    # of the NaN and -inf/+inf cells
    print("Constructing the equity curve...")
    portfolio['returns'] = portfolio['total'].pct_change()
    portfolio['returns'].fillna(0.0, inplace=True)
    portfolio['returns'].replace([np.inf, -np.inf], 0.0, inplace=True)
    portfolio['returns'].replace(-1.0, 0.0, inplace=True)

    # Calculate the full equity curve
    portfolio['returns'] = (portfolio['returns'] + 1.0).cumprod()
    return portfolio
    
if __name__ == "__main__":
    symbols = ('SPY', 'IWM')

    lookbacks = [100,110]
    entry_line = [2.0,1.8,1.6,1.4]
    exit_line = [1.0,0.8,0.6,0.4]
    returns = []
    bars = Bars(symbols).close()
    maxreturns = 0
    mj = 0
    mk = 0
    mlb = 0
    # Adjust lookback period from 50 to 200 in increments
    # of 10 in order to produce sensitivities
    for lb in lookbacks: 
        for j in entry_line:
            for k in exit_line:
                print("Calculating lookback=%s..." % lb) 
                pairs = bars
                pairs = calculate_spread_zscore(pairs, symbols, lookback=lb)
                pairs = create_long_short_market_signals(pairs, symbols,z_entry_threshold=j, z_exit_threshold=k)
                portfolio = create_portfolio_returns(pairs, symbols)
                if portfolio.ix[-1]['returns']>maxreturns:
                    maxreturns = portfolio.ix[-1]['returns']
                    mj = j
                    mk = k
                    mlb = lb
                    print([lb,j,k,portfolio.ix[-1]['returns']])
                else:
                    pass
    print([mlb,mj,mk,maxreturns])
                

#    print("Plot the lookback-performance scatterchart...")
#    plt.plot(lookbacks, returns, '-o')
#    plt.show()
    
#    print("Plotting the performance charts...")
#    fig = plt.figure()
#    fig.patch.set_facecolor('white')
#    ax1 = fig.add_subplot(211,  ylabel='%s growth (%%)' % symbols[0])
#    (pairs[symbols[0]].pct_change()+1.0).cumprod().plot(ax=ax1, color='r', lw=2.)
#    ax2 = fig.add_subplot(212, ylabel='Portfolio value growth (%%)')
#    portfolio['returns'].plot(ax=ax2, lw=2.)
#    fig.show()