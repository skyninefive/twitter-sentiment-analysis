import csv
import os
import re
import nltk
import scipy
import sklearn.metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
import utils

TRAINING_DATA_PATH = './dataset/train_processed.csv'
TEST_DATA_PATH = './dataset/test_processed.csv'

def printLen(name,arr):
    print(name+': '+str(len(arr)))

def getTrainingAndValData():
    X = []
    y = []
    print('starting to read train and val data')
    with open(TRAINING_DATA_PATH, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            tweet_id, result, tweet = line.split(',')
            X.append(tweet)
            y.append(result)
    printLen('X',X) 
    printLen('Y',y) 
    print('splitting data')
    X_train, X_val, y_train, y_val = sklearn.cross_validation.train_test_split(X,y,test_size=0.20, random_state=42)
    printLen('X_train',X_train) 
    printLen('y_train',y_train) 
    printLen('X_val',X_val) 
    printLen('y_val',y_val) 
    return X_train, X_val, y_train, y_val

def classifier(X_train,y_train):
    print('creating classifier')
    vec = TfidfVectorizer(min_df=5, max_df=0.95, sublinear_tf = True,use_idf = True,ngram_range=(1, 2))
    svm_clf =svm.LinearSVC(C=0.1)
    vec_clf = Pipeline([('vectorizer', vec), ('pac', svm_clf)])
    vec_clf.fit(X_train,y_train)
    return vec_clf

def getTestData():
    X_test = []
    print('getting test data')
    with open(TEST_DATA_PATH, 'r') as csv:
        lines = csv.readlines()
        total = len(lines)
        for i, line in enumerate(lines):
            tweet_id, tweet = line.split(',')
            X_test.append(tweet)
    print('X_test len: '+str(len(X_test)))
    return X_test


def saveResultTofile(res):
    print('saving results to csv')
    predictions = [(str(j), int(res[j]))    
                  for j in range(len(res))]
    utils.save_results_to_csv(predictions, 'newSvm.csv')

def main():
    X_train, X_val, y_train, y_val = getTrainingAndValData()
    vec_clf = classifier(X_train,y_train)
    print('predicting')
    y_pred = vec_clf.predict(X_val)
    print('validation results')
    print(sklearn.metrics.classification_report(y_val, y_pred))  
    X_test = getTestData()
    y_test_pred = vec_clf.predict(X_test)
    saveResultTofile(y_test_pred)
        
if __name__ == "__main__":
    main()