# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 19:51:39 2017

@author: Madan
"""
from __future__ import division
def calculate_emd(y_pred, y_test):
    y_pred = list(y_pred)
    y_test = list(y_test)
    d_pred_prob = {}
    d_test_prob = {}
    total_instances = len(y_test)
    #print total_instances
    #all_classes = set(y_test)
    all_classes = [2, 1, 0, -1, -2]
    C = len(all_classes)
    
    for i in all_classes:
        count1 = y_pred.count(i)
        count2 = y_test.count(i)
        d_pred_prob[i] = count1/total_instances
        d_test_prob[i] = count2/total_instances
    #print d_pred_prob
    #print d_test_prob
    score = 0;
    for i in range(0,C):
        sum1 = 0;
        sum2 = 0;
        for j in range(0,i+1):
            sum1 += d_pred_prob[all_classes[j]]
            sum2 += d_test_prob[all_classes[j]]
        score += abs(sum1-sum2)
    return score