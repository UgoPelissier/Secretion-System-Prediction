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
from joblib import Memory
import time

location = './cache'
memory = Memory(location, verbose=0)

start = time.time()

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
    if(False):
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
    return p_true
    
def index_true_false(labels):
    """
    labels: pandas array of labels.csv for the train data set
    
    Returns the indexes of positive and negative labels

    """
    
    labels = labels.values
    index_true = []
    index_false = []
    for i in range(len(labels)):
        if ((labels[i,:].sum())>0):
            index_true.append(i)
        else:
            index_false.append(i)
    return index_true, index_false
    
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
    if(False):
        plt.bar(secretion_system, s)
        plt.xticks(rotation=45)
        plt.show()
    return s, p

def double_proportion(complete_labels):
    """
    complete_labels: pandas array of complete_labels.csv for the train data set  
    
    Returns the proportion of proteins belonging to multiple secretion system families

    """
    
    complete_labels = complete_labels.values
    m = 0
    n = 0
    for i in range(len(complete_labels)):
        if ((complete_labels[i,:].sum())>0):
            m += 1
            if (len((np.where(complete_labels[i,:]>0)[0]))>1):
                n += 1
    print("p_double_all =",round(n/len(complete_labels),3))
    print("p_double_true =",round(n/m,3))
 
load_labels = memory.cache(load_labels)
labels = load_labels()
p_true = labels_proportion(labels)

load_complete_labels = memory.cache(load_complete_labels)
complete_labels = load_complete_labels()

sec_sys, index = secretion_system_list(complete_labels)
s_complete = sum_complete_labels(complete_labels)
s, p = sum_labels(sec_sys,s_complete,index)
# double_proportion(complete_labels)

#-----------------------------------------------------------------------------#
# IMPORT DEPENDENCIES
#-----------------------------------------------------------------------------#
import random
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix
import scipy.stats as stats

#-----------------------------------------------------------------------------#
# PHASE 2: Predicting protein function from it’s sequence
#-----------------------------------------------------------------------------#
def encode_complete_labels(complete_labels, index, s):
    """
    complete_labels: pandas array of complete_labels.csv for the train data set
    index: array of int - Delimitation of the belonging of the columns to a large family
    s: array of int - Number of proteins belonging to each large secretion system families
    
    Returns
        families: list of tuple of int - The families each protein belongs to (can belong to several families)
        groups: array of int - The family each protein is attributed to (by chosing the lesss represented family)

    """
    
    families = []
    groups = []
    complete_labels = complete_labels.values
    for i in range(len(complete_labels)):
        if ((complete_labels[i,:].sum())>0):
            l = list()
            temp = np.where(complete_labels[i,:] > 0)[0]
            for t in temp:
                for j in range(len(index)-1):
                    if ( (index[j]<=t) and (t<index[j+1]) ):
                        if (j+1) not in l:
                            l.append(j+1)
            if (len(l)==0):
                l.append(0)
            families.append(l)
            temp = [ s[k-1] for k in l]
            groups.append(l[np.argmin(temp)])
        else:
            families.append(0)
            groups.append(0)
    return families, np.array(groups)

def load_features():
    """
    Returns pandas array of features.csv for the train data set

    """
    
    X = pd.read_csv("dataset/partial_dataset_train/features.csv", index_col=0)
    X = X[:30000]
    return X

def group_k_fold_true(X,y,groups,index_true,index_false,p_true):
    """
    X: pandas array - Features matrix
    y: pandas array - Labels
    index_true: list of int- Indexes of true labels
    index_false: list of int- Indexes of false labels
    p_true: float - Proportion of true labels
    
    Returns
        index_true_test: array of list of int - Indexes of the true labels to use for each test set
        index_true_train: array of list of int - Indexes of the true labels to use for each train set
        size_false_test: list of int - Numbers of false labels to complete each test set to respect the initial labels proportion
        size_false_train: list of int - Numbers of false labels to complete each train set to respect the initial labels proportion

    """
    
    X_temp = X.values[index_true]
    y_temp = y.values[index_true]
    groups_temp = groups[index_true]
    
    index_true_test = []
    index_true_train = []
    
    size_false_test = []
    size_false_train = []
    
    logo = LeaveOneGroupOut()
    for i, (train_index, test_index) in enumerate(logo.split(X_temp, y_temp, groups=groups_temp)):
        
        index_true_test.append([index_true[idx] for idx in test_index])
        index_true_train.append([index_true[idx] for idx in train_index])
        
        size_false_test.append(int(len(groups_temp[test_index])/p_true))
        size_false_train.append(len(X)-int(len(groups_temp[test_index])/p_true))
        
    return index_true_test, index_true_train, size_false_test, size_false_train

def group_k_fold_false(X,y,index_false,size_false_test,size_false_train):
    """
    X: pandas array - Features matrix
    y: pandas array - Labels
    index_false: list of int- Indexes of false labels
    size_false_test: list of int - Numbers of false labels to complete each test set to respect the initial labels proportion
    size_false_train: list of int - Numbers of false labels to complete each train set to respect the initial labels proportion


    Returns
        index_false_test: array of list of int - Indexes of the true labels to use for each test set
        index_false_train: array of list of int - Indexes of the true labels to use for each train set

    """
    
    X_temp = X.values[index_false]
    y_temp = y.values[index_false]
    
    index_false_test = []
    index_false_train = []
    
    for i in range(len(size_false_test)):
        start = int(np.array(size_false_test[0:i]).sum())
        end = start + size_false_test[i]
        
        if(end>=len(X_temp)):
            end = end%len(index_false)
            index = [*range(start,len(X_temp))] + [*range(0,end)]
            index_false_test.append([index_false[idx] for idx in index])
            
            index = [*range(end,start)]
            index_false_train.append([index_false[idx] for idx in index])
            start = 0
            
        else:
            index_false_test.append([index_false[idx] for idx in range(start,end)])
        
            index = [*range(0,start)] + [*range(end,len(X_temp))]
            index_false_train.append([index_false[idx] for idx in index])
        
    return index_false_test, index_false_train

def assemble_train_test_index(index_false_test,index_false_train,index_true_test,index_true_train):
    """
    index_false_test: array of list of int - Indexes of the true labels to use for each test set
    index_false_train: array of list of int - Indexes of the true labels to use for each train set
    index_true_test: array of list of int - Indexes of the true labels to use for each test set
    index_true_train: array of list of int - Indexes of the true labels to use for each train set
    
    Returns
        train_index: array of list of int - Indexes for the training sets in the cross-validation iterations
        train_index: array of list of int - Indexes for the testing sets in the cross-validation iterations

    """
    
    train_index = []
    test_index = []
    for i in range(len(index_false_test)):
        if (len(index_true_test[i])>20):
            train_index.append(index_false_train[i] + index_true_train[i])
            test_index.append(index_false_test[i] + index_true_test[i])
    return train_index, test_index

def random_undersample(index_false_train,index_true_train,index_true_test):
    """
    index_false_train: array of list of int - Indexes of the true labels to use for each train set
    index_true_train: array of list of int - Indexes of the true labels to use for each train set
    
    Returns
        undersample_train_index: array of list of int - Indexes for the training sets in the cross-validation iterations

    """
    
    undersample_train_index = []
    for i in range(len(index_true_train)):
        if (len(index_true_test[i])>20):
            rd = random.sample(range(len(index_false_train[i])), len(index_true_train[i]))
            undersample_train_index.append([index_false_train[i][r] for r in rd ] + index_true_train[i])
    return undersample_train_index

def remove_null(X):
    """
    X: pandas array - Features matrix
    
    Remove any feature presetning a null value (not used for now)

    """
    
    return X.loc[:, (X != 0).all(axis=0)]

def z_score(X):
    """
    X: pandas array - Features matrix
    
    Compute the z-score for each protein and each feature

    """
    
    return np.abs(stats.zscore(X))

def remove_outliers(X,y,groups,z,n):
    """
    X: pandas array - Features matrix
    y: pandas array - Labels
    groups: array of int - The family each protein is attributed to (by chosing the lesss represented family)
    z: pandas array - z-score
    n: int - Threshold
    
    Remove a protein as soon as its z-score for any feature is above threshold n

    """
    
    return X[(z<n).all(axis=1)], y[(z<n).all(axis=1)], groups[(z<n).all(axis=1)]

def pipeline(X,y,train_index,test_index):
    
    sc = StandardScaler()
    X = sc.fit_transform(X.values)
    
    X_sparse = csr_matrix(X)
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_svd = svd.fit(X_sparse).transform(X_sparse)
    
    pca = PCA(n_components=400, svd_solver='full')
    X_pca = pca.fit_transform(X)
    
    y = y.values.flatten()
    
    acc_log_reg = []
    for i in range(len(test_index)):
        X_train = X_svd[train_index[i]]
        y_train = y[train_index[i]]

        X_test = X_svd[test_index[i]]
        y_test = y[test_index[i]]
        
        # clf = LogisticRegression(solver='newton-cg', max_iter=1000)
        # clf = clf.fit(X_train, y_train)
        
        clf = SVC()
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        acc_log_reg.append(balanced_accuracy_score(y_test, y_pred))
        
    return np.array(acc_log_reg)

encode_families, groups = encode_complete_labels(complete_labels, index, s)    

load_features = memory.cache(load_features)
X = load_features()
y = labels[:30000]
groups = groups[:30000]
z = z_score(X)

X,y,groups = remove_outliers(X,y,groups,z,n=4)
p_true = labels_proportion(y)

index_true, index_false = index_true_false(y)
index_true_test, index_true_train, size_false_test, size_false_train = group_k_fold_true(X, y, groups, index_true, index_false, p_true)
index_false_test, index_false_train = group_k_fold_false(X,y,index_false,size_false_test,size_false_train)
train_index, test_index = assemble_train_test_index(index_false_test,index_false_train,index_true_test,index_true_train)
undersample_train_index = random_undersample(index_false_train,index_true_train,index_true_test)

acc_log_reg = pipeline(X,y,undersample_train_index,test_index)

end = time.time()
print('\nThe function took {:.2f}s to compute.'.format(end - start))
