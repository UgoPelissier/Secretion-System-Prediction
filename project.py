# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 19:44:08 2022

@author: ugo.pelissier
"""

#-----------------------------------------------------------------------------#
# IMPORT DEPENDENCIES
#-----------------------------------------------------------------------------#
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#-----------------------------------------------------------------------------#
# PHASE 1: Preliminary analysis of the dataset and analysis plan
#-----------------------------------------------------------------------------#

def load_labels():
    """
    Returns pandas array of labels.csv for the train data set

    """
    
    return pd.read_csv("dataset/partial_dataset_train/labels.csv", index_col=0)

def load_complete_labels():
    """
    Returns pandas array of complete_labels.csv for the train data set

    """
    
    return (pd.read_csv("dataset/partial_dataset_train/complete_labels.csv", index_col=0)).drop('label', axis=1)

def labels_proportion(labels):
    """
    labels: pandas array of labels.csv for the train data set
    
    Returns the bar plot of the proportion of positive and negative labels in the labels.csv of the train data set

    """
    
    p_true = (1*labels["label"]).sum()/len(labels)
    p_false = 1-p_true
    p = [p_true,p_false]
    fig, ax = plt.subplots()
    plt.bar(["True","False"], [p_true,p_false])
    j = 0
    for i in ax.patches:
        plt.text(i.get_x()+0.3, i.get_y()+0.1,
                 str(round(p[j], 3)),
                 fontsize = 12, fontweight ='bold')
        j += 1
    plt.show()
    # print("p_true =",round(p_true,3))
    
def secretion_system_name(column):
    """
    column: string of the column header (secretion system family) in the pandas array
    
    Returns string of the first two letters of the column

    """
    
    name = ""
    i = 0
    while(i<2):
        name+=column[i]
        i+=1
    return name

def secretion_system_list(complete_labels):
    """
    complete_labels: pandas array of complete_labels.csv for the train data set
    
    Returns
        secretion_system: array of str - Large secretion system families names
        index: array of int - Delimitation of the belonging of the columns to a large family
        
    """
    
    column_name = complete_labels.columns
    secretion_system = np.array([])
    index = np.array([])
    i = 0
    for column in column_name:
        temp = secretion_system_name(column)
        if temp not in secretion_system.tolist():
            secretion_system = np.append(secretion_system, temp)
            index = np.append(index, i)
        i += 1
    index = np.append(index, i)
    index = index.astype(int)
    return secretion_system, index

def sum_complete_labels(complete_labels):
    """
    complete_labels: pandas array of complete_labels.csv for the train data set 
    
    Returns
        s: array of int - Number of proteins belonging to each secretion system sub-families (columns of complete_labels.csv)

    """
    
    column_name = complete_labels.columns
    s = np.array([])
    for column in column_name:
        s = np.append(s,(complete_labels[column]).sum())
    return s

def sum_labels(secretion_system, s_complete, index):
    """
    secretion_system: array of str - Large secretion system families names
    s_complete: array of int - Number of proteins belonging to each secretion system sub-families (columns of complete_labels.csv)
    index: array of int - Delimitation of the belonging of the columns to a large family
    
    Returns
        s: array of int - Number of proteins belonging to each large secretion system families
        p: array of float - Proportion of proteins belonging to each large secretion system families

    """
    
    s = np.array([])
    for i in range(len(index)-1):
        temp = 0
        for j in range(index[i],index[i+1]):
            temp += s_complete[j]
        s = np.append(s,temp)
    p = s/s.sum()
    plt.bar(secretion_system, s)
    plt.xticks(rotation=45)
    return s, p

def double_proportion(complete_labels):
    """
    complete_labels: pandas array of complete_labels.csv for the train data set  
    
    Returns
        Proportion of proteins belonging to multiple secretion system families

    """
    
    complete_labels = complete_labels.values
    n = 0
    for i in range(len(complete_labels)):
        if ((complete_labels[i,:].sum())>1):
            n += 1
    print("p_double =",round(n/len(complete_labels),3))
    
labels = load_labels()
labels_proportion(labels)

complete_labels = load_complete_labels()
sec_sys, index = secretion_system_list(complete_labels)
s_complete = sum_complete_labels(complete_labels)
s, p = sum_labels(sec_sys,s_complete,index)
double_proportion(complete_labels)

#-----------------------------------------------------------------------------#
# PHASE 2: Predicting protein function from it’s sequence
#-----------------------------------------------------------------------------#