#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 15:19:38 2017

@author: lilllianpetersen
"""
import numpy as np

def rolling_median(var,window):
    '''var: array-like. One dimension
    window: Must be odd'''
    n=len(var)
    halfW=int(window/2)
    med=np.zeros(shape=(var.shape))
    for j in range(halfW,n-halfW):
        med[j]=np.ma.median(var[j-halfW:j+halfW+1])
     
    for j in range(0,halfW):
        w=2*j+1
        med[j]=np.ma.median(var[j-w/2:j+w/2+1])
        i=n-j-1
        med[i]=np.ma.median(var[i-w/2:i+w/2+1])
    
    return med    