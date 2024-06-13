#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 11:09:01 2020

@author: AlessandroCornacchia
"""

#%%
import ast, json

dict_list = []
with open("/Users/AlessandroCornacchia/Downloads/lb_opt.txt", "r") as f:
    for row in f:
        dictionary = ast.literal_eval(row)
        dict_list.append(dictionary)
    
d_new = {}    
for d in dict_list:
    _lambda = d['lambda']
    _cdf = d['cdf']['desc'].lower()
    _priorities = len(d['x'])+1
    if not _cdf in d_new:    # first insertion for this cdf
        d_new[_cdf] = {}
    if not _lambda in d_new[_cdf]: # first insertion for this load 
        d_new[_cdf][_lambda] = {}
    if not _priorities in d_new[_cdf][_lambda]:
        d_new[_cdf][_lambda][_priorities] = {}

    # insert value
    d['priorities'] = _priorities
    d_new[_cdf][_lambda][_priorities] = d
    
#%%
r = json.dumps(d_new, indent="\t")
with open("/Users/AlessandroCornacchia/Downloads/lb_opt.json", "w") as f:
    f.write(r)