# -*- coding: utf-8 -*-

from Models import *
from Feature_Extraction import *
from Feature_Extraction import Negations

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
##Other
from sklearn.feature_extraction.text import CountVectorizer,HashingVectorizer
import pandas as pd
import nltk
import re
import codecs
from twitterTokenizer import Tokenizer
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
import string

punct = string.punctuation

def PreProcessingTweets(tweet_text): #input tweet_text is list of tweets
    processed_lists=[]
    for i in tweet_text:
        try:
            tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',i) #substituting urls with constant URL
            tweet = re.sub('@[^\s]+','AT_USER',tweet) #substituting user mentions with constant AT_USER
            tweet = re.sub(r'#([^\s]+)',r'\1',tweet)
            processed_lists.append(tweet)
        except TypeError:
            print "Error Occured"
            tweet = re.sub(r'#([^\s]+)',r'\1',i)
            processed_lists.append(tweet)
        except:
            print "Error Occured"
            tweet = re.sub('@[^\s]+','AT_USER',i) #substituting user mentions with constant AT_USER
            tweet = re.sub(r'#([^\s]+)',r'\1',tweet)
            processed_lists.append(tweet)
    return processed_lists
	
def FeatureExtraction(tweet_text_list):
	index=0
	for tweet in tweet_text_list:
        #ngrams = feature_ngrams(tweet)
        #charGrams = feature_char_grams(tweet)
		exclamtionCount = feature_excalamation_count(tweet)
		questionCount = feature_question_count(tweet)
		Exclamation_Question_Count = feature_excalamation_and_question_count(tweet)
		ElongatedCount=feature_elongated_count(tweet)
		Positive_emoticonCount = feature_positive_emoticon_count(tweet)
		Negative_emoticonCount = feature_negative_emoticon_count(tweet)
		emoticon_existsBinary = feature_emoticon_exists_binary(tweet)
		semantic_lexicons = feature_semantic(tweet)
		features.loc[index]=(tweet,exclamtionCount,questionCount,Exclamation_Question_Count,ElongatedCount,0,
						Positive_emoticonCount,Negative_emoticonCount,emoticon_existsBinary, semantic_lexicons)
		index+=1
	ngrams_train = feature_ngrams(tweet_text_list)
	charGrams_train = feature_char_grams(tweet_text_list)
    #features['n-grams'] = ngrams
    #features['char-grams']=charGrams
	return features,ngrams_train,charGrams_train
	
def FeatureExtraction_test(tweet_text_list):
	index=0
	for tweet in tweet_text_list:
        #ngrams = feature_ngrams(tweet)
        #charGrams = feature_char_grams(tweet)
		exclamtionCount = feature_excalamation_count(tweet)
		questionCount = feature_question_count(tweet)
		Exclamation_Question_Count=feature_excalamation_and_question_count(tweet)
		ElongatedCount=feature_elongated_count(tweet)
		Positive_emoticonCount = feature_positive_emoticon_count(tweet)
		Negative_emoticonCount = feature_negative_emoticon_count(tweet)
		emoticon_existsBinary = feature_emoticon_exists_binary(tweet)
		semantic_lexicons = feature_semantic(tweet)
        features_test.loc[index]=(tweet,exclamtionCount,questionCount,Exclamation_Question_Count,ElongatedCount,0,
                            Positive_emoticonCount,Negative_emoticonCount,emoticon_existsBinary, semantic_lexicons)
        index+=1
    #ngrams_train = feature_ngrams(tweet_text_list)
    #charGrams_train = feature_char_grams(tweet_text_list)
    #features['n-grams'] = ngrams
    #features['char-grams']=charGrams
	return features_test
	
def FeatureExtraction_final_test(tweet_text_list):
	index=0
	for tweet in tweet_text_list:
        #ngrams = feature_ngrams(tweet)
        #charGrams = feature_char_grams(tweet)
		exclamtionCount = feature_excalamation_count(tweet)
		questionCount = feature_question_count(tweet)
		Exclamation_Question_Count=feature_excalamation_and_question_count(tweet)
		ElongatedCount=feature_elongated_count(tweet)
		Positive_emoticonCount = feature_positive_emoticon_count(tweet)
		Negative_emoticonCount = feature_negative_emoticon_count(tweet)
		emoticon_existsBinary = feature_emoticon_exists_binary(tweet)
		semantic_lexicons = feature_semantic(tweet)
		final_features_test.loc[index]=(tweet,exclamtionCount,questionCount,Exclamation_Question_Count,ElongatedCount,0,
                            Positive_emoticonCount,Negative_emoticonCount,emoticon_existsBinary, semantic_lexicons)
		index+=1
    #ngrams_train = feature_ngrams(tweet_text_list)
    #charGrams_train = feature_char_grams(tweet_text_list)
    #features['n-grams'] = ngrams
    #features['char-grams']=charGrams
	return final_features_test
    
def feature_ngrams(tweet_text):   #method to get ngrams (n = [1,2,3,4,..]) of a tweet in taining set
    ngram = HashingVectorizer(strip_accents='unicode', binary=True, ngram_range=(1,4), stop_words=None, lowercase=True,  tokenizer=tokenizer.tokenize, n_features=10000)
    ngram_features = ngram.fit_transform(tweet_text)
    return ngram_features
    #return list(nltk.ngrams(tweet_text,n))
def feature_ngrams_test(tweet_text):
    ngram = HashingVectorizer(strip_accents='unicode', binary=True, ngram_range=(1,4), stop_words=None, lowercase=True,  tokenizer=tokenizer.tokenize, n_features=10000)
    ngram_features = ngram.fit_transform(tweet_text)
    return ngram_features
def feature_char_grams(tweet_text):   #method to get ngrams (n = [1,2,3,4,..]) of a tweet in a training set
    char_gram = HashingVectorizer(strip_accents='unicode', binary=True, ngram_range=(4,5), stop_words=None, lowercase=True, analyzer='char', tokenizer=tokenizer.tokenize, n_features=22000)
    char_gram_features = char_gram.fit_transform(tweet_text)
    return char_gram_features
def feature_charGrams_test(tweet_text):
    char_gram = HashingVectorizer(strip_accents='unicode', binary=True, ngram_range=(4,5), stop_words=None, lowercase=True, analyzer='char', tokenizer=tokenizer.tokenize, n_features=22000)
    char_gram_features = char_gram.fit_transform(tweet_text)
    return char_gram_features
def feature_excalamation_count(tweet_text): #method to get frequency of exclamation(!) in a tweet
    return tweet_text.count("!")
    
def feature_question_count(tweet_text):  #method to get frequency of question(?) in a tweet
    return tweet_text.count("?")
    
def feature_excalamation_and_question_count(tweet_text):#method to get frequency of question mark(?) & exclamation (!) in a tweet
    return tweet_text.count("?") + tweet_text.count("!")
    
def feature_elongated_count(tweet_text): #method to get frequency of elongated words and capitalized words
    count_capitalized=0
    count_elongated=0
    for i in tweet_text.split():
        if i.isupper():
            count_capitalized+=1
        pattern = re.compile(r'(.)\1?')
        result = [x.group() for x in pattern.finditer(i)]
        filtered_result = [x for x in result if len(x) == 2]
        if len(filtered_result) > 2:
            count_elongated+=1
        return count_capitalized+count_elongated
def feature_positive_emoticon_count(tweet_text): #method to get frequency of positive emoticons in a tweet
    emoticons_pos = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # nose
      [\)\]dD\}@]                # mouth      
      |                          # reverse order now! 
      [\)\]dD\}@]                # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
    )"""
    capture_pos_emoticon=re.compile(emoticons_pos,re.VERBOSE|re.I|re.UNICODE)
    capture_pos_emoticon_list=capture_pos_emoticon.findall(tweet_text)
    return len(capture_pos_emoticon_list)
    
def feature_negative_emoticon_count(tweet_text): #method to get frequency of negative emoticons in a tweet
    emoticons_neg = r"""
    (?:
      [<>]?
      [:;=8]                     # eyes
      [\-o\*\']?                 # optional nose
      [\(\[pP/\:\{\|] # mouth      
      |                          # reverse order now! 
      [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
      [\-o\*\']?                 # optional nose
      [:;=8]                     # eyes
      [<>]?
      )"""
    capture_neg_emoticon=re.compile(emoticons_neg,re.VERBOSE|re.I|re.UNICODE)
    capture_neg_emoticon_list=capture_neg_emoticon.findall(tweet_text)
    return len(capture_neg_emoticon_list)
 #method which checks the existence of emoticon in a tweet and returns a binary
def feature_emoticon_exists_binary(tweet_text):
    if feature_negative_emoticon_count(tweet_text) or feature_positive_emoticon_count(tweet_text):
        return 1
    else:
        return 0
		
def feature_semantic(tweet):
	f = []
	tokenized_tweet = tweet.split()
	#NRC Lexicons Uni and Bigrams
	f += nrc_lexicons.get_features(tokenized_tweet)
	
	#Bing_Liu Lexicon	
	#[no_of_positive_words, no_of_negative_words]
	f += bing_liu.get_features(tokenized_tweet)

	#MPQA_SUb_Lexicon
	
	#print mpqa.get_features(tokenized_tweet)
	f += mpqa.get_features(tokenized_tweet)
	#print f
	
	#Find 1000 clusters
	#f += clusters.get_features(tokenized_tweet)
	
	#Negation words
	f += negations.get_features(tokenized_tweet)
	
	return f
	
def DataMatrix(ngram_features,character_gram_features,general_features,categories):
    Final_Features = scipy.sparse.hstack((ngram_features, 
                                          character_gram_features,
                                          general_features.as_matrix(columns=["exclamation_count","question_count",
                                                                              "exclamation_question_count","capital_and_elongated_count",
                                                                              "positve_emoticon_count","negative_emoticon_count","emoticon_exists_binary", "semantic_lexicons"])), dtype=float)
    y=[]
    for i in range(len(categories)):
        if categories[i]=='positive':
            y.append(1)
        elif categories[i] == 'negative':
            y.append(-1)
        elif categories[i] == 'UNKNOWN':
            y.append(0)
        else:
            print categories[i]
    Final_Features= normalize(Final_Features)
    return Final_Features, y

def KLD(true, pred):
    epsilon = 0.5 / len(pred)
    countsTrue, countsPred = Counter(true), Counter(pred)
    p_pos = countsTrue[0]/len(true)
    p_neg = countsTrue[1]/len(true)
    est_pos = countsPred[0]/len(true)
    est_neg = countsPred[1]/len(true)
    p_pos_s = (p_pos + epsilon)/(p_pos+p_neg+2*epsilon)
    p_neg_s = (p_neg + epsilon)/(p_pos+p_neg+2*epsilon)
    est_pos_s = (est_pos+epsilon)/(est_pos+est_neg+2*epsilon)
    est_neg_s = (est_neg+epsilon)/(est_pos+est_neg+2*epsilon)
    return p_pos_s*math.log10(p_pos_s/est_pos_s)+p_neg_s*math.log10(p_neg_s/est_neg_s)

def showMyKLD(true, pred, l):
    s= []
    for key, val in enumerate(l):
        if key == len(l)-1:
            break
        s.append(KLD(true[val:l[key+1]], pred[val:l[key+1]]))
    return sum(s)/len(s)

tokenizer = Tokenizer()
#Load Train data
data = pd.data = pd.read_csv("../Data/twitter-2016train-A.txt", sep="\t", header=None, names=['tweet_id', 'class', 'tweet_text'])

#Load Dev Data
dev_data = pd.data = pd.read_csv("../Data/twitter-2016dev-A.txt", sep="\t", header=None, names=['tweet_id', 'class', 'tweet_text'])

#Load Dev Test
gold_test_data = pd.data = pd.read_csv("../Data/twitter-2016devtest-A.txt", sep="\t", header=None, names=['tweet_id', 'class', 'tweet_text'])

#Final Test Data
final_test_data = pd.data = pd.read_csv("../Data/twitter-2016test-A.txt", sep="\t", header=None, names=['tweet_id', 'class', 'tweet_text'])


#Cleaning & PreProcessing Data
data = data.append(dev_data,ignore_index=True)

data['tweet_text']=PreProcessingTweets(data['tweet_text'])
gold_test_data['tweet_text']=PreProcessingTweets(gold_test_data['tweet_text'])
final_test_data['tweet_text']=PreProcessingTweets(final_test_data['tweet_text'])

#Creating DataFrames for train, dev and test data
features = pd.DataFrame(columns=["tweet","exclamation_count","question_count",
"exclamation_question_count","capital_and_elongated_count","negated_context_count",
"positve_emoticon_count","negative_emoticon_count","emoticon_exists_binary", "semantic_lexicons"])

features_test = pd.DataFrame(columns=["tweet","exclamation_count","question_count",
"exclamation_question_count","capital_and_elongated_count","negated_context_count",
"positve_emoticon_count","negative_emoticon_count","emoticon_exists_binary", "semantic_lexicons"])

final_features_test = pd.DataFrame(columns=["tweet","exclamation_count","question_count",
"exclamation_question_count","capital_and_elongated_count","negated_context_count",
"positve_emoticon_count","negative_emoticon_count","emoticon_exists_binary", "semantic_lexicons"])



#GET FEATURES FOR TRAINING
generalFeatures,nGram_features_train,charGram_features_train = FeatureExtraction(data['tweet_text'])
nGram_features_train.data **= 0.9 #a-power transformation
charGram_features_train.data **= 0.9 #a-power transformation

generalFeatures_test = FeatureExtraction_test(gold_test_data['tweet_text'])
nGram_features_test = feature_ngrams_test(gold_test_data['tweet_text'])
charGram_features_test = feature_charGrams_test(gold_test_data['tweet_text'])
nGram_features_test.data **= 0.9
charGram_features_test.data **= 0.9

generalFeatures_final_test = FeatureExtraction_final_test(final_test_data['tweet_text'])
nGram_features_final_test = feature_ngrams_test(final_test_data['tweet_text'])
charGram_features_final_test = feature_charGrams_test(final_test_data['tweet_text'])
nGram_features_final_test.data **= 0.9
charGram_features_final_test.data **= 0.9

x_train, y_train = DataMatrix(nGram_features_train, 
                              charGram_features_train, 
                              generalFeatures,
                              data['label']) #Combine all  features (train)
x_test,y_test = DataMatrix(nGram_features_test,
                           charGram_features_test,
                           generalFeatures_test,
                           gold_test_data['label'])
                           
x_final_test,y_final_test = DataMatrix(nGram_features_final_test,
                           charGram_features_final_test,
                           generalFeatures_final_test,
                           final_test_data['label'])

from sklearn.metrics import f1_score						   

for c in np.logspace(0,10): #used 100 for submission
	clf = svm.LinearSVC(C=c, loss='squared_hinge', penalty='l2', class_weight='balanced', multi_class='crammer_singer', max_iter=4000, dual=True, tol=1e-6)
	clf.fit(x_train, y_train)
	print f1_score(y_test, clf.predict(x_test), average='micro')