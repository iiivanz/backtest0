# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 23:37:58 2016

@author: Ge
"""
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

p_list = [1,130,3,5,2,46,56,6,4,39]
B = pd.DataFrame(columns=[['ETF','name','class','type']])
def g(x):
    url = 'http://www.etfreplay.com/summary/' + str(x) + '.aspx'
    r = requests.get(url)
    soup = BeautifulSoup(r.text,"lxml")
    cla_list = []
    syb_list = []
    match_list = []
    name_list = []
    title = soup.title.get_text()
    for tag in soup.find_all(id="catheader"):
        cla_list.append(tag.get_text())        
    for tag in soup.find_all(id=re.compile("etfdata_[0-9]{1,}_lblSymbol_[0-9]{1,}")):
        syb_list.append(tag.get_text())
        match_list.append(int(re.findall('[0-9]{1,}', tag['id'])[0]))
    for tag in soup.find_all(id=re.compile("etfdata_[0-9]{1,}_etflink_[0-9]{1,}")):
        name_list.append(tag.get_text())
    A = pd.DataFrame(columns=[['ETF','name','match','class']])
    A['ETF'] = syb_list
    A['name'] = name_list
    A['match'] = match_list
    for y in A.index:
        A['class'][y] = cla_list[A['match'][y]]
    del A['match']
    A['type'] = title
    return A
#B = pd.concat([B,g(x)],ignore_index=True)
    
    
    
