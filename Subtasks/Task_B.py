# -*- coding: utf-8 -*-
from __future__ import division
from Models import *
from Feature_Extraction import *
from Feature_Extraction import twitterTokenizer
from Feature_Extraction import Negations
from PreProcessing import PreProcessingTweets

##Other
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer
import pandas as pd
import nltk
import re
import codecs
#from twitterTokenizer import Tokenizer
from collections import Counter
import sys, subprocess, scipy, numpy as np, os, tempfile, math
from scipy.sparse import csr_matrix
from sklearn.preprocessing import MultiLabelBinarizer
from collections import OrderedDict
from sklearn.cross_validation import train_test_split, cross_val_score
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix


#Constructing Semantic Lexicons
nrc_lexicons = NRC_Lexicon.NRC_Lexicon()
nrc_lexicons.construct_lexicons()

bing_liu = Bing_Liu_Lexicon.Bing_Liu_Lexicon()
bing_liu.construct_lexicons()

mpqa = MPQA_Sub_Lexicon.MPQA_Sub_Lexicon()
mpqa.construct_lexicons()

clusters = Word_Clusters.Word_Clusters()
clusters.construct_clusters()

negations = Negations.Negations()
negations.construct_lexicons()
	
def DataMatrix(ngram_features,character_gram_features,general_features,categories):
    Final_Features = scipy.sparse.hstack((ngram_features, 
                                          character_gram_features,
                                          general_features.as_matrix(columns=["exclamation_count","question_count",
        "exclamation_question_count","capital_and_elongated_count",
        "positve_emoticon_count","negative_emoticon_count","emoticon_exists_binary"]+list("ABCDEFGHIJKL"))), dtype=float)
    y=[]
    for i in range(len(categories)):
        if categories[i]=='positive':
            y.append(1)
        elif categories[i] == 'negative':
            y.append(-1)
        else:
            print categories[i]
    Final_Features= normalize(Final_Features)
    return Final_Features, y


tokenizer = twitterTokenizer.Tokenizer()
#Load Train data
data = pd.data = pd.read_csv("Data/twitter-2016train-BD.txt", sep="\t", header=None, names=['tweet_id','topic' ,'class', 'tweet_text'])

#Load Dev Data
dev_data = pd.data = pd.read_csv("Data/twitter-2016dev-BD.txt", sep="\t", header=None, names=['tweet_id','topic' ,'class', 'tweet_text'])

#Load Dev Test
gold_test_data = pd.read_csv("Data/twitter-2016devtest-BD.txt", sep="\t", header=None, names=['tweet_id','topic' ,'class', 'tweet_text'])

#Final Test Data
final_test_data = pd.read_csv("Data/twitter-2016test-BD.txt", sep="\t", header=None, names=['tweet_id','topic' ,'class', 'tweet_text', 'nan'])

print "Tweet files are loaded"

#Cleaning & PreProcessing Data
data = data.append(dev_data,ignore_index=True)

data['tweet_text']=PreProcessingTweets(data['tweet_text'])
gold_test_data['tweet_text']=PreProcessingTweets(gold_test_data['tweet_text'])
final_test_data['tweet_text']=PreProcessingTweets(final_test_data['tweet_text'])

#Creating DataFrames for train, dev and test data
features = pd.DataFrame(columns=["tweet","exclamation_count","question_count",
"exclamation_question_count","capital_and_elongated_count","negated_context_count",
"positve_emoticon_count","negative_emoticon_count","emoticon_exists_binary"]+list("ABCDEFGHIJKL"))

features_test = pd.DataFrame(columns=["tweet","exclamation_count","question_count",
"exclamation_question_count","capital_and_elongated_count","negated_context_count",
"positve_emoticon_count","negative_emoticon_count","emoticon_exists_binary"]+list("ABCDEFGHIJKL"))

final_features_test = pd.DataFrame(columns=["tweet","exclamation_count","question_count",
"exclamation_question_count","capital_and_elongated_count","negated_context_count",
"positve_emoticon_count","negative_emoticon_count","emoticon_exists_binary"]+list("ABCDEFGHIJKL"))

print "Extracting Features"

#GET FEATURES FOR TRAINING
generalFeatures,nGram_features_train,charGram_features_train = FeatureExtraction(data['tweet_text'], features, nrc_lexicons, bing_liu, mpqa, negations)
nGram_features_train.data **= 0.9 #a-power transformation
charGram_features_train.data **= 0.9 #a-power transformation

generalFeatures_test = FeatureExtraction_test(gold_test_data['tweet_text'], features_test, nrc_lexicons, bing_liu, mpqa, negations)
nGram_features_test = feature_ngrams_test(gold_test_data['tweet_text'])
charGram_features_test = feature_charGrams_test(gold_test_data['tweet_text'])
nGram_features_test.data **= 0.9
charGram_features_test.data **= 0.9

generalFeatures_final_test = FeatureExtraction_final_test(final_test_data['tweet_text'], final_features_test, nrc_lexicons, bing_liu, mpqa, negations)
nGram_features_final_test = feature_ngrams_test(final_test_data['tweet_text'])
charGram_features_final_test = feature_charGrams_test(final_test_data['tweet_text'])
nGram_features_final_test.data **= 0.9
charGram_features_final_test.data **= 0.9

print "Features are ready \nBuilding Data Matrix"
x_train, y_train = DataMatrix(nGram_features_train, 
                              charGram_features_train, 
                              generalFeatures,
                              data['class']) #Combine all  features (train)
x_test,y_test = DataMatrix(nGram_features_test,
                           charGram_features_test,
                           generalFeatures_test,
                           gold_test_data['class'])
                           
x_final_test,y_final_test = DataMatrix(nGram_features_final_test,
                           charGram_features_final_test,
                           generalFeatures_final_test,
                           final_test_data['class'])

						   
print "Classifying"
#for c in np.logspace(0,10,num=10): #used 100 for submission
for c in [1,0.5]:
    clf = svm.LinearSVC(C=c, loss='squared_hinge', penalty='l2', class_weight='balanced', multi_class='crammer_singer', max_iter=4000, dual=True, tol=1e-6)
    clf.fit(x_train, y_train)
    print "C : ",c," ",f1_score(y_test, clf.predict(x_test), average='micro')
    print "Macro-averaged Rcall",score(y_test, clf.predict(x_test))
    print "accuracy: ",accuracy_score(y_test, clf.predict(x_test))
    
print "FINAL TEST SUBMISSION"
#for c in np.logspace(0,10,num=10): #used 100 for submission
for c in [1, 0.5]:
    clf = svm.LinearSVC(C=c, loss='squared_hinge', penalty='l2', class_weight='balanced', multi_class='crammer_singer', max_iter=4000, dual=True, tol=1e-6)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_final_test)
    print "C : ",c," ",f1_score(y_final_test, y_predict, average='micro')
    print "Macro-averaged Recall",score(y_final_test, y_predict)
    print "accuracy: ",accuracy_score(y_final_test, y_predict)
    
final_test_data["predicted"]=y_predict