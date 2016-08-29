# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 16:01:03 2016
@author: Ge
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

data = pd.read_excel("D:\\data.xlsx")
P = data
name = P.iloc[0]
P = P.iloc[2:]
P = P.set_index(["fund_code"])
Pa = P.tail(500)
Par = Pa.pct_change()